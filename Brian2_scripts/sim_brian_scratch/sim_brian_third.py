from brian2 import *
start_scope()
prefs.codegen.target = "numpy"

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
P = PoissonGroup(10, np.arange(10)*Hz + 50*Hz)
G = NeuronGroup(10, equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms )
S = Synapses(P, G, 'w = 1 : 1',on_pre = on_pre, method='linear', delay = 1*ms)
S.connect(j='i')

m1=StateMonitor(G,'v',record=9)
M = StateMonitor(G, 'I', record=9)
m2 = SpikeMonitor(P)

run(300*ms)
print(m2.i)

run(100*ms)

fig1 = plt.figure(figsize=(10,4))
subplot(121)
plot(m1.t/ms,m1.v[0],'-b', label='Neuron 9')
legend()

subplot(122)
plot(M.t/ms, M.I[0], label='I')
legend()
show()