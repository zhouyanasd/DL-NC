from brian2 import *
start_scope()

n1 =2
n2 =2

duration = 1*second
tau = 10*ms
eqs = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''

exc = NeuronGroup(n1, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method='linear')

inh = NeuronGroup(n2, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method='linear')