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


# --- parameter settings ---
Encoding = ['N']
Readout = ['N']
Reservoir = ['N', 'L_1', 'L_2']
Block_random = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P']
Block_scale_free = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'C']
Block_circle = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P_f', 'P_b', 'P_d', 'D_c']
Block_hierarchy = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P_f', 'P_b', 'P_d', 'D_c', 'L']

config_group = ['Encoding', 'Reservoir', 'Block_random', 'Block_scale_free',
                'Block_circle', 'Block_hierarchy', 'Readout']

config_key = {Encoding, Reservoir, Block_random, Block_scale_free,
              Block_circle, Block_hierarchy, Readout}
config_SubCom = [[38], [0, 1, 2], [3, 4, 5, 6, 7, 8, 9],
                 [10, 11, 12, 13, 14, 15, 16],
                 [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                 [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37], [39]]
config_codes = [[None], [None, 1, 1], [None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None, None, None, None], [None]]
config_ranges = [[[0, 1]], [[0, 1]] * 3, [[0, 1]] * 7,
                 [[0, 1]] * 7,
                 [[0, 1]] * 10,
                 [[0, 1]] * 11, [[0, 1]]]
config_borders = [[[0, 1]], [[0, 1]] * 3, [[0, 1]] * 7,
                  [[0, 1]] * 7,
                  [[0, 1]] * 10,
                  [[0, 1]] * 11, [[0, 1]]]
config_precisions = [[0], [4, 0, 0], [4] * 7,
                     [4] * 7,
                     [4] * 10,
                     [4] * 11, [0]]
config_scales = [[0], [0] * 3, [0] * 7,
                 [0] * 7,
                 [0] * 10,
                 [0] * 11, [0]]
