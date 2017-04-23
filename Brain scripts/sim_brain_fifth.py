from brian2 import *
start_scope()
np.random.seed(101)

n = 3

equ = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*20 : 1
'''

on_pre ='''
h+=w
g+=w
'''

P = PoissonGroup(2, 50*Hz)
G = NeuronGroup(n, equ, threshold='v > 0.5', reset='v = 0', method='linear',refractory=1*ms )
S = Synapses(P, G, 'w : 1',on_pre = on_pre, method='linear', delay = 1*ms)
S.connect(j='k for k in range(n)')

S.w = 'rand()'

print(S.w)

m1=StateMonitor(G,('v','I'),record=True)
m2 = SpikeMonitor(G)

run(300*ms)

print(m2.t/ms, m2.i)

fig1 = plt.figure(figsize=(20,8))
subplot(231)
plot(m1.t/ms,m1.v[0],'-b', label='')

subplot(234)
plot(m1.t/ms, m1.I[0], label='I')

subplot(232)
plot(m1.t/ms,m1.v[1],'-b', label='')

subplot(235)
plot(m1.t/ms, m1.I[1], label='I')

subplot(233)
plot(m1.t/ms,m1.v[2],'-b', label='')

subplot(236)
plot(m1.t/ms, m1.I[2], label='I')
show()