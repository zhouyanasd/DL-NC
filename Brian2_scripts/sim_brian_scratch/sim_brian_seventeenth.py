from brian2 import *
start_scope()

taupre = taupost = 15*ms
wmax = 1
Apre = 0.01
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

P = PoissonGroup(2, rates = np.array([15,35])*Hz)

G = NeuronGroup(5, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup')

S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post= on_post_STDP, method='linear', name = 'synapses')

# S.connect(i=0, j=1)
# S.connect(i=1, j=1)
S.connect()

S.w = '0.3+i*'+str(0.3)

M = StateMonitor(S, ['w', 'apre', 'apost'], record=True)
M_G = StateMonitor(G, ['v','I'], record = True)

net = Network(collect())
net.store('first')
S.w = '0.3+i*'+str(0.3)

fig1 = figure(figsize=(20, 8))
subplot(211)
plot(M.t/ms, M.apre[0], label='apre')
plot(M.t/ms, M.apost[0], label='apost')
legend()
subplot(212)
plot(M.t/ms, M.w.T, label='w')
legend(loc='best')
xlabel('Time (ms)')

fig2 = figure(figsize=(20, 8))
subplot(211)
plot(M_G.t/ms, M_G.v[1], label='v 1')
legend()
subplot(212)
plot(M_G.t/ms, M_G.I[1], label='I 1')
legend()
show()