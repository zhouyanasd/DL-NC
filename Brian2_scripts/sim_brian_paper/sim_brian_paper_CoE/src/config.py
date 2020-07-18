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


# --- parameter settings ---
# Encoding = {}
# Readout = {}
Reservoir = ['N', 'L_1', 'L_2']
Block_random = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P']
Block_scale_free = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'C']
Block_circle = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P_f', 'P_b', 'P_d', 'D_c']
Block_hierarchy = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P_f', 'P_b', 'P_d', 'D_c', 'L']

config_key = {'Reservoir': Reservoir, 'Block_random': Block_random, 'Block_scale_free': Block_scale_free,
              'Block_circle': Block_circle, 'Block_hierarchy': Block_hierarchy}
config_SubCom = {'Reservoir': [0, 1, 2], 'Block_random': [3, 4, 5, 6, 7, 8, 9],
                 'Block_scale_free': [10, 11, 12, 13, 14, 15, 16],
                 'Block_circle': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                 'Block_hierarchy': [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]}
config_codes = {'Reservoir': [None, 1, 1], 'Block_random': [None, None, None, None, None, None, None],
                'Block_scale_free': [None, None, None, None, None, None, None],
                'Block_circle': [None, None, None, None, None, None, None, None, None, None, None],
                'Block_hierarchy': [None, None, None, None, None, None, None, None, None, None, None, None]}
config_ranges = {'Reservoir': [[0, 1]] * 3, 'Block_random': [[0, 1]] * 7,
                 'Block_scale_free': [[0, 1]] * 7,
                 'Block_circle': [[0, 1]] * 10,
                 'Block_hierarchy': [[0, 1]] * 11}
config_borders = {'Reservoir': [[0, 1]] * 3, 'Block_random': [[0, 1]] * 7,
                  'Block_scale_free': [[0, 1]] * 7,
                  'Block_circle': [[0, 1]] * 10,
                  'Block_hierarchy': [[0, 1]] * 11}
config_precisions = {'Reservoir': [4, 0, 0], 'Block_random': [4] * 7,
                     'Block_scale_free': [4] * 7,
                     'Block_circle': [4] * 10,
                     'Block_hierarchy': [4] * 11}
config_scales = {'Reservoir': [0] * 3, 'Block_random': [4] * 7,
                 'Block_scale_free': [0] * 7,
                 'Block_circle': [0] * 10,
                 'Block_hierarchy': [0] * 11}