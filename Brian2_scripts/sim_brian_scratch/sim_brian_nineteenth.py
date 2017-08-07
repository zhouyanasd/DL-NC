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

def binary_classification(duration,start=1, end =7, neu =1, interval_l=5,interval_s = ms):
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
    label = np.random.randint(start,end,n)
    seq = tran_bin(label)
    times = where(seq ==1)[0]*interval_s
    indices = zeros(int(len(times)))
    P = SpikeGeneratorGroup(neu, indices, times)
    return P , label

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

###############################################
#-----parameter and model setting-------
n = 20
duration = 500 * ms
duration_test = 200*ms
interval_l = 8
interval_s = ms
threshold = 0.65
obj = 2

taupre = taupost = 2.5*ms
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.2

equ = '''
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = (g-h)*10 : 1
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

#-----simulation setting-------
P, label = binary_classification(duration)
G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup')
G2 = NeuronGroup(round(n/4), equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup_1')
G_readout = NeuronGroup(n,equ_1,method='linear')

# S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post= on_post_STDP, method='linear', name = 'synapses')
S = Synapses(P, G,'w : 1', on_pre=on_pre, method='linear', name = 'synapses')
S3 = Synapses(P, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_2')

S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_1')
# S5 = Synapses(G, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_1')

S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post = on_post_STDP, method='linear',  name = 'synapses_3')
S6 = Synapses(G2, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_1')
S_readout=Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')
# S4 = Synapses(G, G,'w : 1', on_pre=on_pre, method='linear',  name = 'synapses_3')

#-------network topology----------
S.connect(j='k for k in range(n)')
S2.connect()
S3.connect()
S4.connect(condition='i != j', p=0.15)

S.w = '0.1+j*'+str(1/n)
S2.w = '-rand()'
S3.w = '0.3+j*0.2'
S4.w = 'rand()'

#------monitor----------------
m1 = StateMonitor(G_readout, ('I'), record=True, dt = interval_l*interval_s)
m_w = StateMonitor(S, 'w', record=True)
m_w2 = StateMonitor(S4, 'w', record=True)
m_s = SpikeMonitor(P)

###############################################
#------run for pre-train----------------
net = Network(collect())
net.store('first')
net.run(duration)

#------plot the weight----------------
fig2 = plt.figure(figsize= (10,8))
subplot(211)
plot(m_w.t/second, m_w.w.T)
xlabel('Time (s)')
ylabel('Weight / gmax')
subplot(212)
plot(m_w2.t/second, m_w2.w.T)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()

#-------change the synapse model----------
S4.pre.code = '''
h+=w
g+=w
'''
S4.post.code = ''

#-------save the weight----------
net.store('second')
net.restore('first')
S4.w = net._stored_state['second']['synapses_3']['w'][0]
net.store('third')
net.run(duration)

#----lms_train------
obj1 = label_to_obj(label,obj)
m1.record_single_timestep()
Data,para = readout(m1.I,obj1)

#####################################
#----run for test--------
net.restore('third')
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

#------vis of results----------------
print(mse(Y_test,Y_test_t))

# fig0 = plt.figure(figsize=(20, 4))
# plot(mon_s.t/ms, mon_s.i, '.k')
# ylim(-0.5,1.5)

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
show()
