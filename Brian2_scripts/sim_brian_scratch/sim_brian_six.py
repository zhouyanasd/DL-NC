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
    print(M[0])
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

#-----parameter setting-------
n = 50
time_window = 10*ms
duration = 5000 * ms

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

#-----simulation setting-------
P = PoissonGroup(1, 50 * Hz)
G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms)
S = Synapses(P, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
# S2 = Synapses(G[1:3], G[1:3], 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
# S3 = Synapses(G[0:1], G[1:3], 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
# S4 = Synapses(G[1:3], G[0:1], 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
#-------network topology----------
S.connect(j='k for k in range(n)')
# S2.connect(condition='i!=j')
# S3.connect()
# S4.connect()
#
S.w = '0.1+j*'+str(1/n)
# S2.w = 'rand()*2'
# S3.w = '-rand()/5'
# S4.w = 'rand()'
# print('S.w: ',S.w)
# print('S2.w: ',S2.w)
# print('S3.w: ',S3.w)
# print('S4.w: ',S4.w)


#------run----------------
m1 = StateMonitor(G, ('v', 'I'), record=True)
M = []
for i in range(G._N):
    locals()['M' + str(i)] = PopulationRateMonitor(G[(i):(i + 1)])
    M.append(locals()['M' + str(i)])
m6 = PopulationRateMonitor(P)

run(duration)

#----lms_readout----#
#
Z = (m6.smooth_rate(window='gaussian', width=time_window)/ Hz)**2-2000

Data, para = readout(M,Z)
print(para)
Z_t = lms_test(Data,para)
err = abs(Z_t-Z)/max(abs(Z_t-Z))

#------vis----------------
fig1 = plt.figure(figsize=(20, 8))
subplot(231)
plot(m1.t / ms, m1.v[1], '-b', label='v')

subplot(234)
plot(m1.t / ms, m1.I[1], label='I')

subplot(232)
plot(m1.t / ms, m1.v[4], '-b', label='v')

subplot(235)
plot(m1.t / ms, m1.I[4], label='I')

subplot(233)
plot(m1.t / ms, m1.v[8], '-b', label='v')

subplot(236)
plot(m1.t / ms, m1.I[8], label='I')

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
