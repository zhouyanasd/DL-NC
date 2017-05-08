from brian2 import *
from scipy.optimize import leastsq

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

#-----parameter setting-------
n = 80
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
S.w = 'rand()'
# S2.w = 'rand()*2'
# S3.w = '-rand()/5'
# S4.w = 'rand()'
# print('S.w: ',S.w)
# print('S2.w: ',S2.w)
# print('S3.w: ',S3.w)
# print('S4.w: ',S4.w)


#------run----------------
m1 = StateMonitor(G, ('v', 'I'), record=True)
m2 = SpikeMonitor(G)
m3 = PopulationRateMonitor(G[0:1])
m4 = PopulationRateMonitor(G[1:2])
m5 = PopulationRateMonitor(G[2:3])
m6 = PopulationRateMonitor(P)
# print(G.state(name='v'))

run(duration)
# print(m2.spike_trains()[0])
# print(G.t)
# print(m2.t/ms, m2.i)
# print('num: ',m2.num_spikes)
# print('count: ', m2.count)
# print('spike_trains: ', m2.spike_trains()[0])
# print('smooth_rate_P: ', rate)

#----lms_readout----
print(m3)
x1 = m3.smooth_rate(window='gaussian', width=time_window)/ Hz
x2 = m4.smooth_rate(window='gaussian', width=time_window)/ Hz
x3 = m5.smooth_rate(window='gaussian', width=time_window)/ Hz
Z = m6.smooth_rate(window='gaussian', width=time_window)/ Hz

Y = np.zeros((50000))
Y[150:49999] = Z[0:49849]
Z = Y

Data = [x1,x2,x3]
p0=[1,1,1,0.1]
k1,k2,k3,b = lms_train(p0,Z,Data)
para = [k1,k2,k3,b]
print('k1= ',k1,'k2= ', k2, 'k3 = ', k3,'b = ', b)
Z_t = lms_test(Data,para)

#------vis----------------
fig1 = plt.figure(figsize=(20, 8))
subplot(231)
plot(m1.t / ms, m1.v[0], '-b', label='')

subplot(234)
plot(m1.t / ms, m1.I[0], label='I')

subplot(232)
plot(m1.t / ms, m1.v[1], '-b', label='')

subplot(235)
plot(m1.t / ms, m1.I[1], label='I')

subplot(233)
plot(m1.t / ms, m1.v[2], '-b', label='')

subplot(236)
plot(m1.t / ms, m1.I[2], label='I')

fig2 = plt.figure(figsize=(20, 10))
subplot(411)
plot(m3.t / ms, x1,label='neuron1' )
subplot(412)
plot(m4.t / ms, x2,label='neuron2' )
subplot(413)
plot(m5.t / ms, x3,label='neuron3')
subplot(414)
plot(m6.t / ms, Z,'-b', label='Z')
plot(m6.t / ms, Z_t,'-r', label='Z_t')
xlabel('Time (ms)')
ylabel('rate')
show()




# def get_state(G, M, width):
#     spike_trains = M.spike_trains()
#     T = int(G.t/us)
#     n = len(G.i)
#     dt = G.clock.dt
#     rate_G_ = np.zeros((n, T))
#     rate_G = np.zeros((n, T))
#     for g in range(n):
#         for spike in spike_trains[g]:
#             print('fuck: ',spike /ms* 10)
#             rate_G_[g][int(spike /ms* 10)] = (10000 / 3)
#     for g in range(n):
#         width_dt = int(np.round(2 * width / dt))
#         window = np.exp(-np.arange(-width_dt,
#                                    width_dt + 1) ** 2 *
#                         1. / (2 * (width / dt) ** 2))
#         rate_G[g] = Quantity(np.convolve(spike_trains[g],
#                                          window * 1. / sum(window),
#                                          mode='same'), dim=hertz.dim)
#     return rate_G