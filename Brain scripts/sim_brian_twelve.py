from brian2 import *
from scipy.optimize import leastsq
import scipy as sp

start_scope()
np.random.seed(100)

#------define function------------
def lms_train(p0,Zi,Data):
    def error(p, y, args):
        l = len(p)
        f = p[l - 1]
        for i in range(len(args)):
            f += p[i] * args[i]
        return f - y
    Para = leastsq(error,p0,args=(Zi,Data))
    return Para[0]

def lms_test(Data, p):
    l = len(p)
    f = p[l - 1]
    for i in range(len(Data)):
        f += p[i] * Data[i]
    return f

def readout(M,Z):
    n = len(M)
    Data=[]
    for i in M:
        Data.append(i)
    p0 = [1]*n
    p0.append(0.1)
    para = lms_train(p0, Z, Data)
    return Data,para

def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))


def Tri_function(duration):
    rng = np.random
    TIME_SCALE = defaultclock.dt
    in_number = int(duration/TIME_SCALE)

    def sin_fun(l, c, t):
        return (np.sin(c * t * TIME_SCALE/us) + 1) / 2

    def tent_map(l, c, t):
        temp = l
        if (temp < 0.5 and temp > 0):
            temp = (c / 101+1) * temp
            return temp
        elif (temp >= 0.5 and temp < 1):
            temp = (c / 101+1) * (1 - temp)
            return temp
        else:
            return 0.5

    def constant(l, c, t):
        return c / 100

    def chose_fun():
        c = rng.randint(0, 3)
        if c == 0:
            return sin_fun, c
        elif c == 1:
            return tent_map, c
        elif c == 2:
            return constant, c

    def change_fun(rate):
        fun = rng.randint(1, 101)
        if fun > 100 * rate:
            return False
        else:
            return True

    data = []
    cla = []
    cons = rng.randint(1, 101)
    fun, c = chose_fun()

    for t in range(in_number):
        if change_fun(0.7) and t % 50 ==0:
            cons = rng.randint(1, 101)
            fun, c = chose_fun()
            try:
                data_t= fun(data[t - 1], cons, t)
                data.append(data_t)
                cla.append(c)
            except IndexError:
                data_t = fun(rng.randint(1, 101)/100, cons, t)
                data.append(data_t)
                cla.append(c)
        else:
            try:
                data_t = fun(data[t - 1], cons, t)
                data.append(data_t)
                cla.append(c)
            except IndexError:
                data_t= fun(rng.randint(1, 101)/100, cons, t)
                data.append(data_t)
                cla.append(c)
    cla = np.asarray(cla)
    data = np.asarray(data)
    return data, cla

def label_to_obj(label,obj):
    temp = []
    for a in label:
        if a == obj:
            temp.append(1)
        else:
            temp.append(0)
    return np.asarray(temp)

def classification(thea, data):
    def normalization_min_max(arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr))/(np.max(arr)- np.min(arr))
            arr_n[i] = x
        return arr_n
    data_n = normalization_min_max(data)
    data_class = []
    for a in data_n:
        if a >=thea:
            b = 1
        else:
            b = 0
        data_class.append(b)
    return np.asarray(data_class),data_n

#-----parameter setting-------
n = 20
time_window = 5*ms
duration = 200 * ms
interval_l = 8
interval_s = ms
threshold = 0.6

equ = '''
dv/dt = (I-v) / (0.3*ms) : 1 (unless refractory)
dg/dt = (-g)/(0.15*ms) : 1
dh/dt = (-h)/(0.145*ms) : 1
I = (g-h)*40 +I_0: 1
I_0 = stimulus(t)*w_g:1
w_g : 1
'''

equ_1 = '''
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = (g-h)*30 : 1
'''

on_pre = '''
h+=w
g+=w
'''

#-----simulation setting-------

data , label = Tri_function(duration)
stimulus = TimedArray(data,dt=defaultclock.dt)

G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms)
G2 = NeuronGroup(2, equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms)
G_readout=NeuronGroup(n,equ_1,method='linear')

S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
S4 = Synapses(G, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
S_readout=Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

#-------network topology----------
S2.connect()
S4.connect(condition='i != j', p=0.1)
S_readout.connect(j='i')

#
G.w_g = '0.2+i*'+str(0.8/n)
G2.w_g = '0.3+i*0.3'

S2.w = '-rand()/2'
S4.w = 'rand()'

#------run----------------
m3 = StateMonitor(G_readout, ('I'), record=True)
m4 = StateMonitor(G, ('I'), record=True)

run(duration)

#----lms_readout----#
obj1 = label_to_obj(label,2)
Data,para = readout(m3.I,obj1)
print(para)
obj1_t = lms_test(Data,para)
obj1_t_class,data_n = classification(threshold,obj1_t)

#------vis----------------
fig0 = plt.figure(figsize=(20, 4))
plot(data, 'r')

fig1 = plt.figure(figsize=(20, 4))
subplot(111)
plt.scatter(m3.t / ms, obj1_t_class,s=2, color="red", marker='o')
plt.scatter(m3.t / ms, obj1,s=2,color="blue",marker='*')
plt.plot(m3.t / ms, data_n,color="green")
axhline(threshold, ls='--', c='r', lw=1)

fig2 = plt.figure(figsize=(20, 8))
subplot(511)
plot(m3.t / ms, m3.I[1], '-b', label='I')
subplot(512)
plot(m3.t / ms, m3.I[3], '-b', label='I')
subplot(513)
plot(m3.t / ms, m3.I[5], '-b', label='I')
subplot(514)
plot(m3.t / ms, m3.I[7], '-b', label='I')
subplot(515)
plot(m3.t / ms, m3.I[9], '-b', label='I')

fig3 = plt.figure(figsize=(20, 8))
subplot(511)
plot(m4.t / ms, m4.I[1], '-b', label='I')
subplot(512)
plot(m4.t / ms, m4.I[3], '-b', label='I')
subplot(513)
plot(m4.t / ms, m4.I[5], '-b', label='I')
subplot(514)
plot(m4.t / ms, m4.I[7], '-b', label='I')
subplot(515)
plot(m4.t / ms, m4.I[9], '-b', label='I')
show()
