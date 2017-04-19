from brian2 import *
start_scope()

equ = '''
dv/dt = -v / (10*ms) : 1
'''

P = PoissonGroup(100, np.arange(100)*Hz + 10*Hz)
G = NeuronGroup(100, equ)
S = Synapses(P, G, on_pre='v+=0.1')
S.connect(j='i')

m1=StateMonitor(G,'v',record=90)

run(100*ms)

plot(m1.t/ms,m1.v[0],'-b', label='Neuron 90')
show()