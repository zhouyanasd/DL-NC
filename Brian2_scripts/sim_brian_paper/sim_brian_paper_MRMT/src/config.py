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

dynamics_reservoir_synapse_STDP = '''
strength : 1
tau_plasticity : 1
type : 1
dapre/dt = -apre/(taupre*tau_plasticity) : 1 (clock-driven)
dapost/dt = -apost/(taupost*tau_plasticity) : 1 (clock-driven)
'''

dynamics_reservoir_synapse_pre_STDP = '''
I += A_strength_reservoir * strength * property_pre
apre += Apre * (wmax-strength)**type
strength = clip(strength+apost*Switch, wmin, wmax)
'''

dynamics_reservoir_synapse_post_STDP = '''
apost += Apost * (strength-wmin)**type
strength = clip(strength+apre*Switch, wmin, wmax)
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
                    'components_2': {'name':'small_world_2','p_0':[0.3, 0.8], 'p_1':[0.3, 0.8], 'p_2':[0.3, 0.7]}}
Block_max = 20

# --- parameter settings ---
Reservoir_config = ['tau_I', 'type', 'strength', 'tau_plasticity', 'p_connection']
Reservoir_arc = ['arc_connections_'+str(x) for x in range(Block_max)]
Block_0 = ['block', 'N', 'tau', 'tau_I', 'type', 'strength', 'tau_plasticity', 'p_0', 'p_1', 'p_2']
Block_1 = ['block', 'N', 'tau', 'tau_I', 'type', 'strength', 'tau_plasticity', 'p_0', 'p_1', 'p_2']
Block_2 = ['block', 'N', 'tau', 'tau_I', 'type', 'strength', 'tau_plasticity', 'p_0', 'p_1', 'p_2']

config_group = ['Reservoir_config', 'Block_0', 'Block_1', 'Block_2', 'Block_3', 'Encoding_Readout']

config_keys = [Reservoir_config, Block_0, Block_1, Block_2]

config_codes = [[None, None, None, None, None, None],
                [1] * Block_max
                [1, None, None, None, None, None, None, None, None, None],
                [1, None, None, None, None, None, None, None, None, None],
                [1, None, None, None, None, None, None, None, None, None]]

config_ranges = [[[0.1, 1.0], [0, 255], [0, 1], [0.0001, 1.0], [0.0001, 1.0], [0.001, 0.9]],
                 [[0, 1048575]] * Block_max
                 [[0, 3], [15, 300], [0.1, 1.5], [0.1, 1.0], [0, 1], [0.0001, 1.0], [0.0001, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                 [[0, 3], [15, 300], [0.1, 1.5], [0.1, 1.0], [0, 1], [0.0001, 1.0], [0.0001, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                 [[0, 3], [15, 300], [0.1, 1.5], [0.1, 1.0], [0, 1], [0.0001, 1.0], [0.0001, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]

config_borders = [[[0, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1]],
                  [[1, 1]] * Block_max
                  [[1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                  [[1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]]

config_precisions = [[1, 0, 0, 8, 2, 8],
                     [0] * Block_max
                     [0, 0, 2, 1, 0, 8, 2, 8, 8, 8],
                     [0, 0, 2, 1, 0, 8, 2, 8, 8, 8],
                     [0, 0, 2, 1, 0, 8, 2, 8, 8, 8]]

config_scales = [[0] * 6,
                 [0] * Block_max
                 [0] * 10,
                 [0] * 10,
                 [0] * 10]

# '''
# All gen is in float.
# Example:
#
# '''