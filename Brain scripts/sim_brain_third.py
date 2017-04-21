from brian2 import *
start_scope()

equ = '''
dv/dt = (I*15-v) / (20*ms) : 1
I  :1
'''

s_equ ='''
dg/dt = (-g)/(10*ms) : 1 (clock-driven)
dh/dt = (-h)/(9.5*ms) : 1 (clock-driven)
s = g-h : 1
'''

on_pre ='''
h+=1
g+=1
'''
P = PoissonGroup(10, np.arange(10)*Hz + 10*Hz)
G = NeuronGroup(10, equ, threshold='v > 0.9', reset='v = 0', method='linear')
S = Synapses(P, G, model = s_equ, on_pre = on_pre, method='linear')
S.connect(j='i')

G.I = S.s
m1=StateMonitor(G,'v',record=9)
M = StateMonitor(S, 's', record=9)

run(300*ms)

fig1 = plt.figure(figsize=(10,4))
subplot(121)
plot(m1.t/ms,m1.v[0],'-b', label='Neuron 9')

subplot(122)
plot(M.t/ms, M.s[0], label='s')
show()