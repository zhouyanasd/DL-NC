#------------------------------
# Change the Poison input while network is running
#------------------------------
from brian2 import *

prefs.codegen.target = "numpy"  #it is faster than use default "cython"
start_scope()
np.random.seed(15)

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

#---------------------------------------------

equ = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*40 : 1
'''

on_pre ='''
h+=w
g+=w
'''

P = PoissonGroup(5, np.arange(5)*Hz + 50*Hz)
G = NeuronGroup(5, equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms,name = 'neurongroup')
S = Synapses(P, G, 'w : 1',on_pre = on_pre, method='linear',name ='synapses')
S.connect(j =4, i= 4)
S.connect(j =3, i= 3)

S.delay = '0.1*j*ms'
S.w = '0.1+j*0.1'

m1 = StateMonitor(G,('v','I'),record=True)

net = Network(collect())
net.run(1*ms)
net.store('first')
net.run(100*ms)

#-------- change P ------------
net.remove(P)
P2 = PoissonGroup(5, np.arange(5)*Hz + 500*Hz, name = 'poissongroup')
net.add(P2)

S.source = P2
S.pre.source = P2
S._dependencies.remove(P.id)
S.add_dependency(P2)
#------------------------------
net.run(100*ms)


fig1 = plt.figure(figsize=(10,4))
subplot(121)
plot(m1.t/ms,m1.v[4],'-b', label='Neuron 5 V')
legend()

subplot(122)
plot(m1.t/ms, m1.I[4], label='Neuron 5 I')
legend()

show()


