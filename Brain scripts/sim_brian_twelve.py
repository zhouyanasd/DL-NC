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
        Data.append(i[1:])
    p0 = [1]*n
    p0.append(0.1)
    para = lms_train(p0, Z, Data)
    return Data,para

def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def binary_classification(neu =1, interval_l=5,interval_s = ms):
    def tran_bin(A):
        trans = []
        for a in A:
            for i in range(2):
                trans.append(0)
            a_ = bin(a)[2:]
            while len(a_) <3:
                a_ = '0'+a_
            for i in a_:
                trans.append(int(i))
            for i in range(3):
                trans.append(0)
        return np.asarray(trans)
    n = int((duration/interval_s)/interval_l)
    label = np.random.randint(1,3,n)
    seq = tran_bin(label)
    times = where(seq ==1)[0]*interval_s
    indices = zeros(int(len(times)))
    P = SpikeGeneratorGroup(neu, indices, times)
    return P , label


def Tri_function(duration):
    rng = np.random
    TIME_SCALE = defaultclock.dt
    in_number = duration/TIME_SCALE

    def sin_fun(l, c, t):
        return (np.sin(c * t * TIME_SCALE) + 1) / 2

    def tent_map(l, c, t):
        temp = l
        if (temp < 0.5 and temp >= 0):
            temp = (c / 50) * temp
            return temp
        elif (temp >= 0.5 and temp <= 1):
            temp = (c / 50) * (1 - temp)
            return temp
        else:
            return 0

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

    def trans_cls(cla):
        cla_tran = []
        for i in range(3):
            cla_t = np.zeros(cla.size)
            for j in range(cla.size):
                if cla[j] == i:
                    cla_t[j] = 1
                else:
                    cla_t[j] = -1
            cla_tran.append(cla_t)
        return cla_tran

    data_t = np.zeros(in_number)
    cla_t = np.zeros(in_number)
    cons = rng.randint(1, 101)
    fun, c = chose_fun()

    for t in range(in_number):
        if change_fun(0.05):
            cons = rng.randint(1, 101)
            fun, c = chose_fun()
            try:
                data_t[t] = fun(data_t[t - 1], cons, t)
                cla_t[t] = c
            except IndexError:
                data_t[t] = fun(rng.randint(0, 101), cons, t)
                cla_t[t] = c
        else:
            try:
                data_t[t] = fun(data_t[t - 1], cons, t)
                cla_t[t] = c
            except IndexError:
                data_t[t] = fun(rng.randint(0, 101), cons, t)
                cla_t[t] = c
    return data_t, cla_t

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
n = 10
time_window = 5*ms
duration = 2000 * ms
interval_l = 8
interval_s = ms
threshold = 0.65

equ = '''
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = (g-h)*40 : 1
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

P , label = binary_classification(1,interval_l,interval_s)

G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms)
G2 = NeuronGroup(2, equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms)
G_readout=NeuronGroup(n,equ_1,method='linear')
S = Synapses(P, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
S3 = Synapses(P, G2, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
S4 = Synapses(G, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
S_readout=Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

#-------network topology----------
S.connect(j='k for k in range(n)')
S2.connect()
S3.connect()
S4.connect(condition='i != j', p=0.1)
S_readout.connect(j='i')

#
S.w = '0.2+j*'+str(0.8/n)
S2.w = '-rand()/2'
S3.w = '0.3+j*0.3'
S4.w = '0'



#------run----------------
m1 = StateMonitor(G_readout, ('I'), record=True, dt = interval_l*interval_s)
m2 = SpikeMonitor(P)
m3 = StateMonitor(G_readout, ('I'), record=True)
m4 = StateMonitor(G, ('I'), record=True)

run(duration)

#----lms_readout----#
obj1 = label_to_obj(label,2)
m1.record_single_timestep()
Data,para = readout(m1.I,obj1)
print(para)
obj1_t = lms_test(Data,para)
obj1_t_class,data_n = classification(threshold,obj1_t)

#------vis----------------
fig0 = plt.figure(figsize=(20, 4))
plot(m2.t/ms, m2.i, '.k')

fig1 = plt.figure(figsize=(20, 4))
subplot(111)
plt.scatter(m1.t[1:] / ms, obj1_t_class,color="red")
plt.scatter(m1.t[1:] / ms, obj1,color="blue")
plt.scatter(m1.t[1:] / ms, data_n,color="green")
axhline(threshold, ls='--', c='r', lw=3)

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
