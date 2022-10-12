from brian2 import defaultclock, ms

# --- dynamic models ---
Dt = defaultclock.dt = 1 * ms
Switch = 1

taupre = taupost = 100*ms
wmin = 0
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05

A_strength_block = 1
A_strength_reservoir = 1
A_strength_encoding = 1

standard_tau = 100 * ms

A_tau_I = 10 * ms

voltage_reset = 0.2

threshold_solid = voltage_reset
threshold_max = 2
threshold_jump = 0.01
A_threshold = 1

refractory_reservoir = 2 * ms

dynamics_encoding = '''
property = 1 : 1
I = stimulus(t,i) : 1
'''

dynamics_reservoir = '''
property : 1
tau : 1
tau_I : 1
dv/dt = (I-v+voltage_reset) / (standard_tau) : 1 (unless refractory)
dI/dt = (-I)/(tau_I * A_tau_I) : 1
dthreshold/dt = (threshold_solid-threshold)/(tau * standard_tau) : 1
'''

dynamics_readout = '''
count : 1
tau_I : 1
dv/dt = (I-v) / (standard_tau) : 1
dI/dt = (-I)/(tau_I * A_tau_I) : 1
'''

dynamics_block_synapse_STDP = '''
strength : 1
tau_plasticity : 1
type : 1
dapre/dt = -apre/(taupre*tau_plasticity) : 1 (clock-driven)
dapost/dt = -apost/(taupost*tau_plasticity) : 1 (clock-driven)
'''

dynamics_block_synapse_pre_STDP = '''
I += A_strength_block * strength * property_pre 
apre += Apre * (wmax-strength)**type
strength = clip(strength+apost*Switch, wmin, wmax)
'''

dynamics_block_synapse_post_STDP = '''
apost += Apost * (strength-wmin)**type
strength = clip(strength+apre*Switch, wmin, wmax)
'''

dynamics_reservoir_synapse = '''
strength : 1
'''

dynamics_reservoir_synapse_pre = '''
I += A_strength_reservoir * strength * property_pre
'''

dynamics_encoding_synapse_STDP = '''
strength : 1
tau_plasticity : 1
type : 1
dapre/dt = -apre/(taupre*tau_plasticity) : 1 (clock-driven)
dapost/dt = -apost/(taupost*tau_plasticity) : 1 (clock-driven)
'''

dynamics_encoding_synapse_pre_STDP = '''
I += A_strength_encoding * strength * property_pre
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

threshold_reservoir = 'v > A_threshold * threshold'

reset_reservoir = '''
v = voltage_reset
threshold = clip(threshold+threshold_jump*Switch, threshold_solid, threshold_max)
'''

# --- block layer and reservoir structure ---
structure_blocks = {'components_0': {'name':'random', 'p_0':[0.01, 0.3], 'p_1':None, 'p_2':None},
                    'components_1': {'name':'scale_free', 'p_0':[0.1, 1.0], 'p_1':[0.1, 1.0], 'p_2':[0.1, 1.0]},
                    'components_2': {'name':'small_world_2', 'p_0':[0.3, 0.8], 'p_1':[0.3, 0.8], 'p_2':[0.3, 0.7]},
                    'components_3': {'name':'three_layer', 'p_0':[0.5, 1.0], 'p_1':[0.5, 1.0], 'p_2':[0.5, 1.0]}}
block_init = 3
block_max = 10
# increase_threshold = 0.05

# --- parameter settings ---
Reservoir_config = ['strength', 'p_connection']
Reservoir_arc = ['connections_' + str(x) for x in range(block_max)]
Block_config = ['block', 'N', 'tau', 'tau_I', 'type', 'strength', 'tau_plasticity', 'p_0', 'p_1', 'p_2']
Encoding_Readout = ['tau_I', 'type', 'strength', 'tau_plasticity', 'p_connection']

config_group_reservoir = ['Reservoir_config', 'Reservoir_arc']
config_group_block = ['Block', 'Encoding_Readout']

config_keys_reservoir = [Reservoir_config, Reservoir_arc]
config_keys_block = [Block_config, Encoding_Readout]

config_SubCom_reservoir = [[0, 1], [x + 2 for x in range(block_max)]]
config_SubCom_block = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]

config_codes_reservoir = [[None, None], [1] * block_init + [None] * (block_max-block_init)]
config_codes_block = [[1, None, None, None, None, None, None, None, None, None], [None, None, None, None, None]]

config_ranges_reservoir = [[[0.0001, 1.0], [0.001, 0.9]],
                           [[0, 2**block_init-1]] * block_init + [[0, 0]] * (block_max-block_init)]
config_ranges_block = [[[0, 3], [100, 300], [0.1, 1.5], [0.1, 1.0], [0, 1], [0.0001, 1.0], [0.0001, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                       [[0.1, 1.0], [0, 1], [0.0001, 1.0], [0.0001, 1.0], [0.001, 0.005]]]

config_borders_reservoir = [[[0, 1], [0, 1]],
                            [[1, 1]] * block_max]
config_borders_block = [[[1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                        [[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]]

config_precisions_reservoir = [[8, 8], [0] * block_max]
config_precisions_block = [[0, 0, 2, 1, 0, 8, 2, 8, 8, 8], [1, 0, 8, 2, 8]]

config_scales_reservoir = [[0] * 2, [0] * block_max]
config_scales_block = [[0] * 10, [0] * 5]

gen_group_reservoir = [[0, 1]]
gen_group_block = [[0, 1]]

'''
All gen is in float.
Example:
gen_reservoir = [0.4, 27.0, 1.0, 0.9, 0.2, 0.6, 
                5768, ... , 14895]
gen_block = [0, 110.0, 0.3, 0.4, 1.0, 0.9, 0.3, 0.22, 0.22, 0.22,
             0.4, 1.0, 0.7, 0.5, 0.7]
'''