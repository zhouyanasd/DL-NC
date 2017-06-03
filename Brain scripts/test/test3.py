from brian2 import *
start_scope()

equ_1 = '''
I = stimulus(t)*w : 1
w  = weight: 1
'''

# Data =np.linspace(0,11,12)
Data =np.linspace(0,11,12).reshape(3,4)
stimulus = TimedArray(Data,dt=1*ms)
weight = 1

G = NeuronGroup(1, equ_1)
M = StateMonitor(G, 'I', record=True)

# G.w = '[1,1,1,1]'

run(2*ms)

print(M.I)