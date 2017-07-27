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

P = PoissonGroup(2, rates=15*Hz)

G = NeuronGroup(2, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup')

S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post= on_post_STDP, method='linear', name = 'synapses')

S.connect(i=0, j=1)

S.w = '0.1+j*'+str(1/2)

M = StateMonitor(S, ['w', 'apre', 'apost'], record=True)

net = Network(collect())
net.store('first')
net.run(5000*ms)

figure(figsize=(20, 8))
subplot(211)
plot(M.t/ms, M.apre[0], label='apre')
plot(M.t/ms, M.apost[0], label='apost')
legend()
subplot(212)
plot(M.t/ms, M.w[0], label='w')
legend(loc='best')
xlabel('Time (ms)')