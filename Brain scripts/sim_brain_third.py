from brian2 import *
start_scope()

equ = '''
dv/dt = (1-v) / (10*ms) : 1
'''

s_equ ='''
dg/dt = (-g)/(10*ms) : 1 (clock-driven)
'''

on_pre ='''
g+=1
v+=g
'''
P = PoissonGroup(10, np.arange(10)*Hz + 30*Hz)
G = NeuronGroup(10, equ, threshold='v > 2.0', reset='v = 0', method='linear')
S = Synapses(P, G, model = s_equ, on_pre = on_pre, method='linear')
S.connect(j='i')

m1=StateMonitor(G,'v',record=9)
M = StateMonitor(S, 'g', record=9)

run(300*ms)

fig1 = plt.figure(figsize=(10,4))
subplot(121)
plot(m1.t/ms,m1.v[0],'-b', label='Neuron 9')

subplot(122)
plot(M.t/ms, M.g[0], label='g')
show()