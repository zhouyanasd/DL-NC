from brian2 import *
start_scope()
prefs.codegen.target = "numpy"  #it is faster than use default "cython"

@check_units(spike_window=1,result=1)
def get_rate(spike_window):
    return np.sum(spike_window, axis = 1)
# get_rate = Function(get_rate, arg_units=[1],
#                     return_unit=1)

@check_units(spike_window=1, spike = 1, result=1)
def get_spike_window(spike_window, spike):
    new_window = np.zeros(spike_window.shape)
    new_window[:,:-1] = spike_window[:,1:]
    new_window[:,-1] = spike
    return new_window


equ_1 = '''
dv/dt = (I-v) / (1.5*ms) : 1 
I = stimulus(t,i): 1
spike : 1
rate : 1
'''

equ_2 = '''
I : 1
d_n : 1
d_t : 1
'''

model='''
w : 1
I_post = w * I_pre : 1 (summed)
d = 1 : 1
'''

on_pre = '''
d_n = 0
d_n += w * I_pre
d_t += d
'''

reset = '''
v = 0
spike = 1
'''

event_input = '''
spike_window = get_spike_window(spike_window, spike)
rate = get_rate(spike_window)
spike = 0
'''

Data =np.linspace(0,11,12).reshape(3,4)
print(Data)
stimulus = TimedArray(Data,dt=1*ms)
print(stimulus(2*ms,2))

Input = NeuronGroup(4, equ_1, events={'input_sin':'t<2*ms'},method='euler',threshold='v > 0.5', reset=reset)
# G = NeuronGroup(2, equ_2)
# S = Synapses(Input, G, model, on_pre= on_pre, on_event={'pre':'input_sin'})

Input.run_on_event('input_sin', event_input)
Input.variables.add_dynamic_array('spike_window', size=(4,5))
Input.spike = 0
Input.spike_window = 0

# S.connect()
# S.w = '1'
# S.delay = 'rand()*ms'

M_input = StateMonitor(Input, ('v','rate'), record=True)
S_input = SpikeMonitor(Input)

run(2*ms)

print('M_input.rate: %s'%M_input.rate)
print('M_input.v: %s'%M_input.v[:])
print(S_input.i, S_input.t, S_input.count)

# for name, var in G.variables.items():
#     print('%r : %s' % (name, var))