from brian2 import *
start_scope()

equ = '''
dv/dt = (1*mV - v) / (10*ms) : volt (unless refractory)
'''

P = PoissonGroup(1, 100*Hz)
G = NeuronGroup(100, 'dv/dt = (1*mV - v) / (10*ms) : volt (unless refractory)',threshold='v > 0.9*mV', reset='v = 0*mV',
                    refractory=0.1*ms, method='linear')
S = Synapses(P, G, on_pre='v+=0.3*mV')
S.connect(j='i')

m1=StateMonitor(G,'v',record=9)

run(200*ms)

plot(m1.t/ms,m1.v[0],'-b', label='Neuron 9')
show()