from brian2 import *
start_scope()

equ = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*20 : 1
'''

equ2 = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*40 : 1
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

net = Network(collect())
net.run(300*ms)
print(S.w)
store("test4","../Data/test4")

# start_scope()
G1 = NeuronGroup(10, equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms )
m_g1=StateMonitor(G,'v',record=9)
net.add(G1)
net.add(m_g1)
print(collect())

net.run(10*ms)

print(m_g1.t)

net.remove(G)
net.remove(S)
net.remove(m1)
net.remove(M)

net.run(10*ms)

print(m1.t)
# P1 = PoissonGroup(10, np.arange(10)*Hz + 50*Hz)
# G1 = NeuronGroup(10, equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms )
# S1 = Synapses(P, G, 'w = 2 : 1',on_pre = on_pre, method='linear', delay = 1*ms)
# S1.connect(j='i')

# restore("test4","../Data/test4")
# run(100*ms)
# print(S1.w)

fig1 = plt.figure(figsize=(10,4))
subplot(121)
plot(m1.t/ms,m1.v[0],'-b', label='Neuron 9 V')
legend()

subplot(122)
plot(M.t/ms, M.I[0], label='Neuron 9 I')
legend()
show()