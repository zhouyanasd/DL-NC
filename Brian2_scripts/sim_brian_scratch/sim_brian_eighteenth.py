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

def readout(M):
    print(M[0])
    n = len(M)
    Data=[]
    for i in range(n):
        x = M[i].smooth_rate(window='gaussian', width=time_window)/ Hz
        Data.append(x)
    return Data

def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def save_para(para, name):
    np.save('../Data/temp/'+str(name)+'.npy',para)

def load_para(name):
    return np.load('../Data/temp/'+str(name)+'.npy')

#-----parameter setting-------
n = 10
time_window = 10*ms
duration = 5000 * ms
duration_test = 2000*ms

taupre = taupost = 15*ms
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.2

equ = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*10 : 1
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
P = PoissonGroup(1, 50 * Hz)
G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup')
G2 = NeuronGroup(round(n/10), equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup_1')
# S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post= on_post_STDP, method='linear', name = 'synapses')
S = Synapses(P, G,'w : 1', on_pre=on_pre, method='linear', name = 'synapses')
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
S4.w = 'rand()'

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
print("w4-be:",S4.w)
net.store('first')
net.run(6000*ms)

#------plot the weight----------------
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

#-------change the synapse model----------
S4.pre.code = '''
h+=w
g+=w
'''
S4.post.code = ''
#-------save the weight----------
import copy
w4 = copy.copy(S4.w)
print("w4-af:",w4)
net.store('second')
net.restore('first')
S4.w = w4
net.run(duration)
print("w4:",S4.w)
#-----test_Data----------
Data = readout(M)
Y = (m_y.smooth_rate(window='gaussian', width=time_window)/ Hz)

#----lms_train------
p0 = [1]*n
p0.append(0.1)
para = lms_train(p0, Y, Data)

#----run for test--------
net.run(duration_test, report='text')
print("w4:",S4.w)
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
