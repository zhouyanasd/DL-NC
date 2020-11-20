from brian2 import *

# --- dynamic models ---
Dt = defaultclock.dt = 1 * ms
Switch = 1

taupre = taupost = 20*ms
wmin = 0
wmax = 1
Apre = 0.2
Apost = -Apre*taupre/taupost*1.05

A_strength = 60
A_strength_reservoir = 120
A_strength_encoding = 200

threshold_solid = 0.5
A_threshold = 1.5

standard_tau = 100
tau_read = 30

voltage_reset = 13.5

refractory_reservoir = 3 * ms

dynamics_encoding = '''
property = 1 : 1
I = stimulus(t,i) : 1
'''

dynamics_reservoir = '''
property : 1
tau : 1
threshold : 1
dv/dt = (I-v) / (tau * standard_tau * ms) : 1 (unless refractory)
dg/dt = (-g)/(3*ms) : 1
I = g + voltage_reset : 1
'''

dynamics_readout = '''
tau = tau_read : 1
dv/dt = (I-v) / (tau*ms) : 1
dg/dt = (-g)/(3*ms) : 1
I = g : 1
'''

dynamics_synapse = '''
strength : 1
'''

dynamics_synapse_pre = '''
g += A_strength * strength * property_pre 
'''

dynamics_synapse_STDP = '''
strength : 1
plasticity : 1
type : 1
dapre/dt = -apre/(taupre*plasticity) : 1 (clock-driven)
dapost/dt = -apost/(taupost*plasticity) : 1 (clock-driven)
'''

dynamics_synapse_pre_STDP = '''
g += A_strength * strength
apre += Apre * (wmax-strength)**type
strength = clip(strength+apost*Switch, wmin, wmax)
'''

dynamics_synapse_post_STDP = '''
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
g += A_strength * strength
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
g += A_strength * strength
apre += Apre * (wmax-strength)**type
strength = clip(strength+apost*Switch, wmin, wmax)
'''

dynamics_encoding_synapse_post_STDP = '''
apost += Apost * (strength-wmin)**type
strength = clip(strength+apre*Switch, wmin, wmax)
'''

threshold_encoding = 'I > 0'

threshold_reservoir = 'v > voltage_reset + threshold_solid + A_threshold * threshold'

reset_reservoir = 'v = voltage_reset'

# --- reservoir layer structure ---
structure_blocks = {'components_0':'random',
                    'components_1':'scale_free',
                    'components_2':'circle',
                    'components_3':'hierarchy'}

structure_layer = {'components_0':{'structure': [[0,0,1],[1,2,2]], 'output_input':[[2],[0]]},
                    'components_1': {'structure':[[0],[1]], 'output_input':[[1],[0]]},
                    'components_2': {'structure':[[0,1],[1,2]], 'output_input':[[2],[0]]},
                    'components_3': {'structure':[[0,0,1,2],[1,3,2,3]], 'output_input':[[3],[0]]}}

structure_reservoir = {'components': {'structure':[[0,1,2],[1,2,3]],'output_input':[[0,1,2,3],[0,1,2,3]]}}

# --- parameter settings ---
Reservoir_config = ['block', 'layer_1', 'layer_2', 'type', 'strength', 'plasticity']
Block_random = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'p']
Block_scale_free = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'p_alpha', 'p_beta', 'p_gama']
Block_circle = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'p_forward', 'p_backward', 'p_threshold']
Block_hierarchy = ['N_i', 'N_h', 'N_o', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'p_out', 'p_in', 'decay']
Encoding_Readout = ['type', 'strength', 'plasticity']

config_group = ['Reservoir_config', 'Block_random', 'Block_scale_free',
                'Block_circle', 'Block_hierarchy', 'Encoding_Readout']

config_keys = [Reservoir_config, Block_random, Block_scale_free,
              Block_circle, Block_hierarchy, Encoding_Readout]


config_SubCom = [[0, 1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10, 11, 12],
                 [13, 14, 15, 16, 17, 18, 19, 20, 21],
                 [22, 23, 24, 25, 26, 27, 28, 29, 30],
                 [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                 [42, 43, 44]]

config_codes = [[1, 1, 1, None, None, None],
                [None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None, None],
                [None, None, None]]

config_ranges = [[[0, 255], [0, 255], [0, 255], [0, 1], [0, 1], [0, 1]],
                 [[15, 30]] + [[0, 1]] * 6,
                 [[15, 30]] +[[0, 1]] * 8,
                 [[15, 30]] +[[0, 1]] * 8,
                 [[5, 10]] + [[5, 10]] + [[5, 10]] + [[0, 1]] * 8,
                 [[0, 1]] * 3]

config_borders = [[[1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[1, 1], [0, 1], [0, 1]]]

config_precisions = [[0, 0, 0, 0, 4, 4],
                     [0, 4, 4, 0, 4, 4, 4],
                     [0, 4, 4, 0, 4, 4, 4, 4, 4],
                     [0, 4, 4, 0, 4, 4, 4, 4, 4],
                     [0, 0, 0, 4, 4, 0, 4, 4, 4, 4, 4],
                     [0, 4, 4]]

config_scales = [[0] * 6,
                 [0] * 7,
                 [0] * 9,
                 [0] * 9,
                 [0] * 11,
                 [0] * 3]

'''
All gen is in float.
Example:
gen = [15.0, 129.0, 251.0, 1.0, 0.9,0.2,
      17.0, 0.3, 0.3, 1.0, 0.9, 0.3, 0.33,
      20.0, 0.4, 0.4, 1.0, 0.4, 0.4, 0.44, 0.44, 0.44,
      23.0, 0.5, 0.5, 1, 0.5, 0.5, 0.55, 0.55, 0.55,
      6.0, 8.0, 9.0, 0.6, 0.6, 1.0, 0.6, 0.6, 0.16, 0.16, 0.16,
      1.0, 0.7, 0.7]
'''