from brian2 import *
start_scope()
prefs.codegen.target = "numpy"  #it is faster than use default "cython"

equ_1 = '''
I = stimulus(t,i): 1
'''

equ_2 = '''
I : 1
d_n : 1
'''

model='''
w : 1
I_post = w * I_pre : 1 (summed)
d = 1 : 1
'''

# Data =np.linspace(0,11,12)
Data =np.linspace(0,11,12).reshape(3,4)
print(Data)
stimulus = TimedArray(Data,dt=1*ms)
print(stimulus(2*ms,2))

Input = NeuronGroup(4, equ_1,events={'input_sin':'1>0'})
G = NeuronGroup(2, equ_2)
S = Synapses(Input, G, model, on_pre= 'd_n += d', on_event={'pre':'input_sin'})

S.connect()
S.w = '1'

M = StateMonitor(G, ('I','d_n'), record=True)

run(2*ms)

print(M.I[:])
print(S.I_pre[:])
print(S.I_post[:])
print(M.d_n)

for name, var in G.variables.items():
    print('%r : %s' % (name, var))