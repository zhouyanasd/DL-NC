from brian2 import *

#------define function------------
def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
#--------------------------------------

start_scope()
np.random.seed(15)

equ = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*10 : 1
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
S = Synapses(P, G, 'w = 1 : 1',on_pre = on_pre, method='linear')
S.connect(j =9, i= 9)
S.connect(j =8, i= 8)
S.delay = 'j*ms'

m1 = StateMonitor(G,'v',record=9)
M = StateMonitor(G, 'I', record=9)
m2 = SpikeMonitor(P)

net = Network(collect())
net.run(100*ms)
visualise_connectivity(S)


G.equations._equations['I'] = "I = (g-h)*30 : 1"
G.equations._equations.pop('I')
G.equations = G.equations+("I = (g-h)*30 : 1")


net.run(100*ms)


fig1 = plt.figure(figsize=(10,4))
subplot(121)
plot(m1.t/ms,m1.v[0],'-b', label='Neuron 9 V')
legend()

subplot(122)
plot(M.t/ms, M.I[0], label='Neuron 9 I')
legend()

show()