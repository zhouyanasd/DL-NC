from brian2 import *

start_scope()
np.random.seed(101)


def lms_train():
    pass

def lms_test():
    pass


n = 3
time_window = 2 * ms
duration = 10 * ms

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

P = PoissonGroup(1, 50 * Hz)
G = NeuronGroup(n, equ, threshold='v > 0.25', reset='v = 0', method='linear', refractory=1 * ms)
S = Synapses(P, G, 'w : 1', on_pre=on_pre, method='linear', delay=1 * ms)
S.connect(j='k for k in range(n)')

S.w = 'rand()'

print(S.w)

m1 = StateMonitor(G, ('v', 'I'), record=True)
m2 = SpikeMonitor(G)
m3 = PopulationRateMonitor(G[0:1])
m4 = PopulationRateMonitor(G[1:2])
m5 = PopulationRateMonitor(G[2:3])
# print(G.state(name='v'))

run(500 * ms)
# print(m2.spike_trains()[0])
# print(G.t)
# print(m2.t/ms, m2.i)
# print('num: ',m2.num_spikes)
# print('count: ', m2.count)
# print('spike_trains: ', m2.spike_trains()[0])
# print('smooth_rate_P: ', rate)

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

fig2 = plt.figure(figsize=(20, 8))
subplot(311)
plot(m3.t / ms, m3.smooth_rate(window='gaussian', width=time_window) / Hz)
subplot(312)
plot(m4.t / ms, m4.smooth_rate(window='gaussian', width=time_window) / Hz)
subplot(313)
plot(m5.t / ms, m5.smooth_rate(window='gaussian', width=time_window) / Hz)
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