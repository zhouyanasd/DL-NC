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
threshold = 0.5

taupre = taupost = 15*ms
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.2

equ = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*20 : 1
'''

on_pre = '''
h+=w
g+=w
'''

model_STDP= '''
w : 1
dapre/dt = -apre/taupre : 1 (clock-driven)
dapost/dt = -apost/taupost : 1 (clock-driven)
'''

on_pre_STDP = '''
h+=w
g+=w
apre += Apre
w = clip(w+apost, 0, wmax)
'''

on_post_STDP= '''
apost += Apost
w = clip(w+apre, 0, wmax)
'''

#-----simulation setting-------P = correlated_data(4)
P = correlated_data(3)

G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup')
G2 = NeuronGroup(round(n/10), equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup_1')
S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post= on_post_STDP, method='linear', name = 'synapses')
# S = Synapses(P, G,'w : 1', on_pre=on_pre, method='linear', name = 'synapses')
S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_1')
S3 = Synapses(P, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_2')
S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post = on_post_STDP, method='linear',  name = 'synapses_3')
# S4 = Synapses(G, G,'w : 1', on_pre=on_pre, method='linear',  name = 'synapses_3')

#-------network topology----------
S.connect(j='k for k in range(n)')
S2.connect()
S3.connect()
S4.connect(condition='i != j', p=0.2)

S.w = '0.1+j*'+str(1/n)
S2.w = '-rand()/2'
S3.w = '0.3+j*0.2'
S4.w = 'rand()/2'

#------monitor----------------
M = []
for i in range(G._N):
    locals()['M' + str(i)] = PopulationRateMonitor(G[(i):(i + 1)])
    M.append(locals()['M' + str(i)])
m_y = PopulationRateMonitor(P)

mon_w = StateMonitor(S, 'w', record=True)
mon_w2 = StateMonitor(S4, 'w', record=True)

mon_s = SpikeMonitor(P)

#------run for pre-train----------------
net = Network(collect())
net.store('first')
net.run(duration)

#-----test_Data----------
Data = readout(M)
Y = (m_y.smooth_rate(window='gaussian', width=time_window)/ Hz)

#----lms_train------
p0 = [1]*n
p0.append(0.1)
para = lms_train(p0, Y, Data)

#-------change the synapse model--------------
S.pre.code = '''
h+=w
g+=w
'''
S.post.code = ''

S4.pre.code = '''
h+=w
g+=w
'''
S4.post.code = ''

#----run for test--------
net.run(duration_test, report='text')

#-----test_Data----------
Data = readout(M)
Y = (m_y.smooth_rate(window='gaussian', width=time_window)/ Hz)

#-----lms_test-----------
Y_t = lms_test(Data,para)
err = (abs(Y_t-Y)/max(Y))

t0 = int(duration/defaultclock.dt)
t1 = int((duration+duration_test) / defaultclock.dt)

Y_test = Y[t0:t1]
Y_test_t = Y_t[t0:t1]
err_test = err[t0:t1]
t_test = m_y.t[t0:t1]

#------vis----------------
print(mse(Y_test,Y_test_t))

fig0 = plt.figure(figsize=(20, 4))
plot(mon_s.t/ms, mon_s.i, '.k')
ylim(-0.5,1.5)

fig1 = plt.figure(figsize=(20, 10))
subplot(311)
plot(m_y.t / ms, Y,'-b', label='Y')
plot(m_y.t / ms, Y_t,'--r', label='Y_t')
xlabel('Time (ms)')
ylabel('rate')
legend()
subplot(312)
plot(m_y.t / ms, err,'-r', label='err')
xlabel('Time (ms)')
ylabel('err')
legend()
subplot(313)
plot(t_test / ms, Y_test,'-b', label='Y_test')
plot(t_test / ms, Y_test_t,'-r', label='Y_test_t')
xlabel('Time (ms)')
ylabel('rate')
legend()

fig2 = plt.figure(figsize= (10,8))
subplot(211)
plot(mon_w.t/second, mon_w.w.T)
xlabel('Time (s)')
ylabel('Weight / gmax')
subplot(212)
plot(mon_w2.t/second, mon_w2.w.T)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()

show()




