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

def readout(M,Y):
    n = len(M)
    Data=[]
    for i in range(n):
        x = M[i].smooth_rate(window='gaussian', width=time_window)/ Hz
        Data.append(x)
    p0 = [1]*n
    p0.append(0.1)
    para = lms_train(p0, Z, Data)
    return Data,para

def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def binary_classification(neu =1, interval_l=8,interval_s = ms):
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
    n = int(duration/(interval_l*interval_s))
    label = np.random.randint(1,8,n)
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
            temp.append(-1)
    return np.asarray(temp)

def I_to_data(mon,interval_l,interval_s):
    dt = duration/(interval_l*interval_s)
    data=[]
    for m in mon.indices:
        data_t = []
        pass

#-----parameter setting-------
n = 10
time_window = 5*ms
duration = 2000 * ms
interval_l = 8
interval_s = ms

equ = '''
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = (g-h)*30 : 1
'''

on_pre = '''
h+=w
g+=w
'''

#-----simulation setting-------
# stimulus = TimedArray(np.tile([100.,0.,100.,0.], 10)*Hz, dt=100.*ms)
# P = PoissonGroup(2, rates='stimulus(t)')

P , label = binary_classification(interval_l,interval_s)

G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms)
G2 = NeuronGroup(2, equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms)
S = Synapses(P, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
S3 = Synapses(P, G2, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
S4 = Synapses(G, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
# S2 = Synapses(G[1:3], G[1:3], 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
# S3 = Synapses(G[0:1], G[1:3], 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
# S4 = Synapses(G[1:3], G[0:1], 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)

#-------network topology----------
S.connect(j='k for k in range(n)')
S2.connect()
S3.connect()
S4.connect(condition='i != j', p=0.1)
# S2.connect(condition='i!=j')
# S3.connect()
# S4.connect()
#
S.w = '0.1+j*'+str(1/n)
S2.w = '-rand()/2'
S3.w = '0.3+j*0.3'
S4.w = 'rand()'
# S2.w = 'rand()*2'
# S3.w = '-rand()/5'
# print('S.w: ',S.w)
# print('S2.w: ',S2.w)
# print('S3.w: ',S3.w)
# print('S4.w: ',S4.w)


#------run----------------
m1 = StateMonitor(G, ('v', 'I'), record=True, dt = interval_l*interval_s)
m2 = SpikeMonitor(P)
M = []
for i in range(G._N):
    locals()['M' + str(i)] = PopulationRateMonitor(G[(i):(i + 1)])
    M.append(locals()['M' + str(i)])
m6 = PopulationRateMonitor(P)

run(duration)

#----lms_readout----#
#
Z = (m6.smooth_rate(window='gaussian', width=time_window)/ Hz)

Data, para = readout(M,Z)
print(para)
Z_t = lms_test(Data,para)
err = abs(Z_t-Z)/max(abs(Z_t-Z))

#----
obj1 = label_to_obj(label,1)

m1.record_single_timestep()

#------vis----------------
fig0 = plt.figure(figsize=(20, 4))
plot(m2.t/ms, m2.i, '.k')

fig2 = plt.figure(figsize=(20, 10))
subplot(511)
plot(M[1].t / ms, Data[1],label='neuron1' )
subplot(512)
plot(M[4].t / ms, Data[4],label='neuron2' )
subplot(513)
plot(M[8].t / ms, Data[8],label='neuron3')
subplot(514)
plot(m6.t / ms, Z,'-b', label='Z')
plot(m6.t / ms, Z_t,'-r', label='Z_t')
xlabel('Time (ms)')
ylabel('rate')
subplot(515)
plot(m6.t / ms, err,'-b', label='Z')
xlabel('Time (ms)')
ylabel('err')
show()
