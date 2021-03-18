from brian2 import defaultclock, ms

# --- dynamic models ---
Dt = defaultclock.dt = 1 * ms
Switch = 1

taupre = taupost = 100*ms
wmin = 0
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05

A_strength = 1
A_strength_reservoir = 1
A_strength_encoding = 0.001

threshold_solid = 0.2
threshold_max = 1
a_threshold = 0.2
A_threshold = 0.01

standard_tau = 100 * ms
tau_I = 1 * ms

voltage_reset = 0.2

refractory_reservoir = 2 * ms

dynamics_encoding = '''
property = 1 : 1
I = stimulus(t,i) : 1
'''

dynamics_reservoir = '''
property : 1
tau : 1
dv/dt = (I-v+voltage_reset) / (tau * standard_tau) : 1 (unless refractory)
dI/dt = (-I)/(tau_I) : 1
dthreshold/dt = (threshold_solid-threshold)/(tau * standard_tau) : 1
'''

dynamics_readout = '''
count : 1
dv/dt = (I-v) / (standard_tau) : 1
dI/dt = (-I)/(tau_I) : 1
'''

dynamics_block_synapse_STDP = '''
strength : 1
plasticity : 1
type : 1
dapre/dt = -apre/(taupre*plasticity) : 1 (clock-driven)
dapost/dt = -apost/(taupost*plasticity) : 1 (clock-driven)
'''

dynamics_block_synapse_pre_STDP = '''
I += A_strength * strength
apre += Apre * (wmax-strength)**type
strength = clip(strength+apost*Switch, wmin, wmax)
'''

dynamics_block_synapse_post_STDP = '''
apost += Apost * (strength-wmin)**type
strength = clip(strength+apre*Switch, wmin, wmax)
'''

dynamics_reservoir_synapse_STDP = '''
strength : 1
plasticity : 1
type : 1
dapre/dt = -apre/(taupre*plasticity) : 1 (clock-driven)
dapost/dt = -apost/(taupost*plasticity) : 1 (clock-driven)
'''

dynamics_reservoir_synapse_pre_STDP = '''
I += A_strength_reservoir * strength
apre += Apre * (wmax-strength)**type
strength = clip(strength+apost*Switch, wmin, wmax)
'''

dynamics_reservoir_synapse_post_STDP = '''
apost += Apost * (strength-wmin)**type
strength = clip(strength+apre*Switch, wmin, wmax)
'''

dynamics_encoding_synapse_STDP = '''
strength : 1
plasticity : 1
type : 1
dapre/dt = -apre/(taupre*plasticity) : 1 (clock-driven)
dapost/dt = -apost/(taupost*plasticity) : 1 (clock-driven)
'''

dynamics_encoding_synapse_pre_STDP = '''
I += A_strength_encoding * strength
apre += Apre * (wmax-strength)**type
strength = clip(strength+apost*Switch, wmin, wmax)
'''

dynamics_encoding_synapse_post_STDP = '''
apost += Apost * (strength-wmin)**type
strength = clip(strength+apre*Switch, wmin, wmax)
'''

dynamics_readout_synapse_pre = '''
I += strength
count += 1
'''

threshold_encoding = 'I > 0'

threshold_reservoir = 'v >= voltage_reset + A_threshold * threshold'

reset_reservoir = '''
v = voltage_reset
threshold = clip(threshold+a_threshold, threshold_solid, threshold_max)
'''

# --- reservoir layer structure ---
structure_blocks = {'components_0':'random',
                    'components_1':'scale_free',
                    'components_2':'circle',
                    'components_3':'hierarchy'}

structure_layer = {'components_0':{'structure': [[],[]], 'output_input':[[0,1,2,3],[0,1,2,3]]},
                    'components_1': {'structure':[[0,0,1,2],[1,2,3,3]], 'output_input':[[3],[0]]},
                    'components_2': {'structure':[[0,0,2,2],[1,3,1,3]], 'output_input':[[1,3],[0,2]]},
                    'components_3': {'structure':[[0,0,0],[1,2,3]], 'output_input':[[1,2,3],[0]]}}

structure_reservoir = {'components': {'structure':[[],[]],'output_input':[[0,1,2,3],[0,1,2,3]]}}

# --- parameter settings ---
Reservoir_config = ['block', 'layer_1', 'type', 'strength', 'plasticity']
Block_random = ['N', 'tau', 'type', 'strength', 'plasticity', 'p']
Block_scale_free = ['N', 'tau', 'type', 'strength', 'plasticity', 'p_alpha', 'p_beta', 'p_gama']
Block_circle = ['N', 'tau', 'type', 'strength', 'plasticity', 'p_forward', 'p_backward', 'p_threshold']
Block_hierarchy = ['N_i', 'N_h', 'N_o', 'tau', 'type', 'strength', 'plasticity', 'p_out', 'p_in', 'decay']
Encoding_Readout = ['type', 'strength', 'p_connection', 'plasticity']

config_group = ['Reservoir_config', 'Block_random', 'Block_scale_free',
                'Block_circle', 'Block_hierarchy', 'Encoding_Readout']

config_keys = [Reservoir_config, Block_random, Block_scale_free,
              Block_circle, Block_hierarchy, Encoding_Readout]


config_SubCom = [[0, 1, 2, 3, 4],
                 [5, 6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15, 16, 17, 18],
                 [19, 20, 21, 22, 23, 24, 25, 26],
                 [27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
                 [37, 38, 39, 40]]

config_codes = [[1, 1, None, None, None],
                [None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None]]

config_ranges = [[[0, 255], [0, 255], [0, 1], [0.1, 1.0], [0.1, 1.0]],
                 [[15, 150], [0.1, 1.0], [0, 1], [0.1, 1.0], [0.1, 1.0], [0.1, 0.3]],
                 [[15, 150], [0.1, 1.0], [0, 1], [0.1, 1.0], [0.1, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                 [[15, 150], [0.1, 1.0], [0, 1], [0.1, 1.0], [0.1, 1.0], [0.1, 0.5], [0.1, 0.5], [0.1, 0.5]],
                 [[5, 50], [5, 50], [5, 50], [0.1, 1.0], [0, 1], [0.1, 1.0], [0.1, 1.0], [0.3, 1.0], [0.3, 1.0], [0.5, 0.9]],
                 [[0, 1], [0.1, 1.0], [0.1, 1.0], [0.1, 1.0]]]

config_borders = [[[1, 1], [1, 1], [1, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[1, 1], [0, 1], [0, 1], [0, 1]]]

config_precisions = [[0, 0, 0, 4, 4],
                     [0, 4, 0, 4, 4, 4],
                     [0, 4, 0, 4, 4, 4, 4, 4],
                     [0, 4, 0, 4, 4, 4, 4, 4],
                     [0, 0, 0, 4, 0, 4, 4, 4, 4, 4],
                     [0, 4, 4, 4]]

config_scales = [[0] * 5,
                 [0] * 6,
                 [0] * 8,
                 [0] * 8,
                 [0] * 10,
                 [0] * 4]

gen_group = [[0], [1], [2, 3, 4, 5]]

'''
All gen is in float.
Example:
gen = [27.0, 27.0, 1.0, 0.9,0.2,
      110.0, 0.3, 1.0, 0.9, 0.3, 0.22,
      100.0, 0.4, 1.0, 0.4, 0.4, 0.44, 0.44, 0.44,
      100.0, 0.5, 1.0, 0.5, 0.5, 0.33, 0.33, 0.33,
      50.0, 30.0, 40.0, 0.6, 1.0, 0.6, 0.6, 0.44, 0.55, 0.66,
      1.0, 0.7, 0.5, 0.7]
'''