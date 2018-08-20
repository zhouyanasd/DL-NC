from brian2 import *
start_scope()
prefs.codegen.target = "numpy"  #it is faster than use default "cython"

@check_units(spike_window=1,result=1)
def get_rate(spike_window):
    return np.sum(spike_window, axis = 1)/spike_window.shape[1]

@check_units(spike_window=1, spike = 1, result=1)
def get_spike_window(spike_window, spike):
    new_window = np.zeros(spike_window.shape)
    new_window[:,:-1] = spike_window[:,1:]
    new_window[:,-1] = spike
    return new_window

learning_rate = 0.02

equ_1 = '''
dv/dt = (I-v) / (1.5*ms) : 1 
I = stimulus(t,i): 1
spike : 1
rate : 1
'''

equ_2 = '''
dv/dt = (I-v) / (1.5*ms) : 1 
dI/dt = (-I) / (0.5*ms) : 1 
spike : 1
rate : 1
'''

model='''
w : 1
dth_m/dt = (rate_post-th_m)/(1.5*ms) : 1 (clock-driven)
'''

on_pre = {
    'pre': '''
    I += w
    ''',
    'pathway_rate':'''
     d_w = rate_pre*(rate_post - th_m)*rate_post - learning_rate*w
     w += d_w
    '''}

reset = '''
v = 0
spike = 1
'''

event_rate = '''
spike_window = get_spike_window(spike_window, spike)
rate = get_rate(spike_window)
spike = 0
'''

Data =np.linspace(0,11,12).reshape(3,4)
stimulus = TimedArray(Data,dt=1*ms)

Input = NeuronGroup(4, equ_1, events={'event_rate':'True'},method='euler',threshold='v > 0.5', reset=reset)
G = NeuronGroup(2, equ_2, events={'event_rate':'True'}, method='euler',threshold='v > 0.5', reset=reset)
S = Synapses(Input, G, model, on_pre= on_pre, on_event={'pre':'spike', 'pathway_rate': 'event_rate'},method='euler')

Input.run_on_event('event_rate', event_rate)
Input.variables.add_dynamic_array('spike_window', size=(4,5))
Input.spike = 0
Input.spike_window = 0

G.run_on_event('event_rate', event_rate)
G.variables.add_dynamic_array('spike_window', size=(4,5))
G.spike = 0
G.spike_window = 0

S.connect()
S.w = '1'

M_g = StateMonitor(G, ('v','rate'), record=True)
S_g = SpikeMonitor(G)
M_input = StateMonitor(Input, ('v','rate'), record=True)
S_input = SpikeMonitor(Input)
M_s = StateMonitor(S, ('w','th_m'), record=True)

run(2*ms)

print('M_input.rate: %s'%M_input.rate)
print('M_g.rate: %s'%M_g.rate)
print('M_s.w: %s'%M_s.w[:])
print('M_s.th_m: %s'%M_s.th_m[:])
print(S_input.i, S_input.t, S_input.count)