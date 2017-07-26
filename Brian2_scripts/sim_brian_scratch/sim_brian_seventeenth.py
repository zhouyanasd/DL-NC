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
n = 1
time_window = 10*ms
duration = 5000 * ms
duration_test = 2000*ms

taupre = taupost = 2*ms
wmax = 1
Apre = 0.02
Apost = -Apre*taupre/taupost*1.5

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

#-----simulation setting-------
P = PoissonGroup(1, 50 * Hz)
G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup')
# G2 = NeuronGroup(round(n/10), equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup_1')
S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post= on_post_STDP, method='linear', name = 'synapses')
# S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_1')
# S3 = Synapses(P, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_2')
# S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post = on_post_STDP, method='linear',  name = 'synapses_3')

#-------network topology----------
S.connect(j='k for k in range(n)')
# S2.connect()
# S3.connect()
# S4.connect(condition='i != j', p=0.1)

S.w = '0.1+j*'+str(1/n)
# S2.w = '-rand()/2'
# S3.w = '0.3+j*0.3'
# S4.w = 'rand()'

#------monitor----------------
M = []
for i in range(G._N):
    locals()['M' + str(i)] = PopulationRateMonitor(G[(i):(i + 1)])
    M.append(locals()['M' + str(i)])
m_y = PopulationRateMonitor(P)

mon_w = StateMonitor(S, 'w', record=True)

mon_s = SpikeMonitor(P)

#------run for pre-train----------------
net = Network(collect())
net.run(duration)
net.store('first')

#-----test_Data----------
Data = readout(M)
Y = (m_y.smooth_rate(window='gaussian', width=time_window)/ Hz)

#----lms_train------
p0 = [1]*n
p0.append(0.1)
para = lms_train(p0, Y, Data)

#-----lms_test-----------
Y_t = lms_test(Data,para)
err = abs(Y_t-Y)

#------vis----------------
fig0 = plt.figure(figsize=(20, 4))
plot(mon_s.t/ms, mon_s.i, '.k')

fig1 = plt.figure(figsize=(20, 10))
subplot(211)
plot(m_y.t / ms, Y,'-b', label='Y')
plot(m_y.t / ms, Y_t,'--r', label='Y_t')
xlabel('Time (ms)')
ylabel('rate')
subplot(212)
plot(m_y.t / ms, err,'-r', label='err')
xlabel('Time (ms)')
ylabel('err')


fig2 = plt.figure(figsize= (10,8))
# subplot(311)
# plot(S.w / wmax, '.k')
# ylabel('Weight / wmax')
# xlabel('Synapse index')
# subplot(312)
# hist(S.w / wmax, 20)
# xlabel('Weight / wmax')
subplot(313)
plot(mon_w.t/second, mon_w.w.T)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
show()

print(mse(Y_t,Y))