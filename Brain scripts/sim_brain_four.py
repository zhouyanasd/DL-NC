from brian2 import *
start_scope()

stimulus = TimedArray(np.hstack([[c, c, c, 0, 0]
                                 for c in np.random.rand(1000)]),
                                dt=10*ms)

equ = '''
dv/dt = (-v + stimulus(t)*mV)/(10*ms) : volt
'''

G = NeuronGroup(1, equ,threshold='v > 0.9*mV', reset='v = 0*mV',
                    refractory=1*ms, method='linear')

m1=StateMonitor(G,'v',record=0)

run(200*ms)

plot(m1.t/ms,m1.v[0],'-b', label='Neuron 0')
show()