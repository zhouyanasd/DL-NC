from brian2 import *

# --- dynamic models ---
dynamics_encoding = '''
property = 1 : 1
I = stimulus(t,i) : 1
'''

dynamics_reservoir = '''
property : 1
tau : 1
dv/dt = (I-v) / (tau*ms) : 1 (unless refractory)
dg/dt = (-g)/(3*ms) : 1
I = g + 13.5 : 1
'''

dynamics_readout = '''
tau : 1
dv/dt = (I-v) / (tau*ms) : 1
dg/dt = (-g)/(3*ms) : 1
I = g : 1
'''

dynamics_synapse = '''
w : 1
'''

dynamics_synapse_pre = '''
g += w * property_pre 
'''

dynamics_synapse_STDP = '''
w : 1
dapre/dt = -apre/taupre : 1 (clock-driven)
dapost/dt = -apost/taupost : 1 (clock-driven)
'''

dynamics_synapse_pre_STDP = '''
h+=w
g+=w
apre += Apre
w = clip(w+apost, wmin, wmax)
'''

dynamics_synapse_post_STDP = '''
apost += Apost
w = clip(w+apre, wmin, wmax)
'''

threshold_encoding = 'I > 0'

threshold_reservoir = 'v > 15'

reset_reservoir = 'v = 13.5'

refractory_reservoir = 3 * ms

# --- reservoir layer structure ---
structure_blocks = {'components_1':'random',
                    'components_2':'scale_free',
                    'components_3':'circle',
                    'components_4':'hierarchy'}

structure_layer = {'components_1':{'structure': [[0,0,1,2],[1,2,3,3]], 'output_input':[[3],[0]]},
                    'components_2': {'structure':[[0,0,0,1,2],[1,2,3,3,3]], 'output_input':[[3],[0]]},
                    'components_3': {'structure':[[0,0,1,1,2],[1,2,3,3,3]], 'output_input':[[3],[0]]},
                    'components_4': {'structure':[[0,0,1,3],[1,2,3,2]], 'output_input':[[2],[0]]}}

structure_reservoir = {'components': {'structure':[[0,1,2],[1,2,3]],'output_input':[[0,1,2,3],[0,1,2,3]]}}

# --- parameter settings ---
Reservoir = ['block', 'layer_1', 'layer_2', 'strength', 'plasticity']
Block_random = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'p']
Block_scale_free = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'p_alpha', 'p_beta', 'p_gama']
Block_circle = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'p_forward', 'p_backward', 'p_threshold']
Block_hierarchy = ['N_i', 'N_h', 'N_o', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'p_out', 'p_in', 'decay']
Encoding_Readout = ['strength', 'plasticity']

config_keys = [Reservoir, Block_random, Block_scale_free,
              Block_circle, Block_hierarchy, Encoding_Readout]

config_group = ['Reservoir', 'Block_random', 'Block_scale_free',
                'Block_circle', 'Block_hierarchy', 'Encoding_Readout']

config_SubCom = [[0, 1, 2, 3, 4],
                 [5, 6, 7, 8, 9, 10, 11],
                 [12, 13, 14, 15, 16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25, 26, 27, 28, 29],
                 [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                 [41, 42]]

config_codes = [[1, 1, 1, None, None],
                [None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None, None],
                [None, None]]

config_ranges = [[[0, 255], [0, 255], [0, 255], [0, 1], [0, 1]],
                 [[10, 90]] + [[0, 1]] * 6,
                 [[10, 90]] +[[0, 1]] * 8,
                 [[10, 90]] +[[0, 1]] * 8,
                 [[10, 30]] + [[10, 30]] + [[10, 30]] + [[0, 1]] * 11,
                 [[0, 1]] * 2]

config_borders = [[[1, 1], [1, 1], [1, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[0, 1]] * 2]

config_precisions = [[0, 0, 0, 4, 4],
                     [4, 4, 4, 0, 4, 4, 4] * 7,
                     [4, 4, 4, 0, 4, 4, 4, 4, 4] * 9,
                     [4, 4, 4, 0, 4, 4, 4, 4, 4] * 9,
                     [4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4] * 11,
                     [4, 4] * 2]

config_scales = [[0] * 5,
                 [0] * 7,
                 [0] * 9,
                 [0] * 9,
                 [0] * 11,
                 [0] * 2]
