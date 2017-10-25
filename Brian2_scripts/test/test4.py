from brian2 import *

prefs.codegen.target = "numpy"  #it is faster than use default "cython"

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
I = (g-h)*0 : 1
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
S = Synapses(P, G, 'w : 1', on_pre = on_pre, method='linear')
S.connect(j =9, i= 9)
S.connect(j =8, i= 8)
S.delay = 'j*ms'
S.w = '0.1+j*0.1'

m1 = StateMonitor(G,'v',record=9)
M = StateMonitor(G, 'I', record=9)
m2 = SpikeMonitor(P)

net = Network(collect())
net.run(100*ms)
visualise_connectivity(S)

# store("test4","../Data/test4")

# start_scope()
# G1 = NeuronGroup(10, equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms )
# m_g1=StateMonitor(G,'v',record=9)
# net.add(m_g1)
from brian2.equations.equations import parse_string_equations
G.equations._equations['I'] = parse_string_equations("I = (g-h)*30 : 1")['I']
G.equations._equations.pop('I')
G.equations = G.equations+("I = (g-h)*40 : 1")
G.variables._variables['I'].expr = '(g-h)*10'
M.variables._variables['_source_I'].expr = '(__source_I_neurongroup_g-__source_I_neurongroup_h)*10'

S.pre.code = on_pre
# net.run(100*ms)


# net.remove(G)
# net.remove(S)
# net.remove(m1)
# net.remove(M)
# net.remove(m_g1)

# G = NeuronGroup(10, equ2, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms, name = 'neurongroup1' )
# print(G.id)
# S1 = Synapses(P, G, 'w = 1 : 1',on_pre = on_pre, method='linear', delay = 1*ms)
# m_g2=StateMonitor(G,'v',record=9)
# net.add(G)
# net.add(S1)
# net.add(m_g2)

# S1.active = False
# print(collect())
net.run(100*ms)
# P1 = PoissonGroup(10, np.arange(10)*Hz + 50*Hz)
# G1 = NeuronGroup(10, equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms )
# S1 = Synapses(P, G, 'w = 2 : 1',on_pre = on_pre, method='linear', delay = 1*ms)
# S1.connect(j='i')

# restore("test4","../Data/test4")
# G.equations._equations['I'] = "I = (g-h)*30 : 1"
# run(100*ms)

# profiling_summary(net,show=5)

fig1 = plt.figure(figsize=(15,4))
subplot(131)
plot(m1.t/ms,m1.v[0],'-b', label='Neuron 9 V')
legend()

subplot(132)
plot(M.t/ms, M.I[0], label='Neuron 9 I')
legend()

# subplot(133)
# plot(m_g2.t/ms, m_g2.v[0], label='Neuron2 9 V')
# legend()
show()