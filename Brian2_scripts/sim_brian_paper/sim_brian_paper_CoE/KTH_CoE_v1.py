# -*- coding: utf-8 -*-
"""
    The Neural Structure Search (NAS) of Liquid State Machine
    (LSM) for action recognition. The optimization method adopted
    here is Cooperative Co-evolution (CoE).

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.

Requirement
=======
Numpy
Pandas
Brian2

Usage
=======

Citation
=======

"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.config import *

from brian2 import *

warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)
data_path = '../../../Data/KTH/'

###################################
# -----simulation parameter setting-------
GenerateData = False
DataName = 'temp'

origin_size = (120, 160)
pool_size = (5, 5)
pool_types = 'max'
threshold = 0.2

F_train = 1
F_validation = 1
F_test = 1
Dt = defaultclock.dt = 1 * ms
standard_tau = 100

# -------class initialization----------------------
math_function = MathFunctions()
base_function = BaseFunctions()
evaluate = Evaluation()
KTH = KTH_classification()

# -------data initialization----------------------
try:
    df_en_train = KTH.load_data(data_path + 'train_' + DataName+'.p')
    df_en_validation = KTH.load_data(data_path + 'validation_' + DataName+'.p')
    df_en_test = KTH.load_data(data_path + 'test_' + DataName+'.p')

    data_train_s, label_train = KTH.get_series_data_list(df_en_train, is_group=True)
    data_validation_s, label_validation = KTH.get_series_data_list(df_en_validation, is_group=True)
    data_test_s, label_test = KTH.get_series_data_list(df_en_test, is_group=True)
except FileNotFoundError:
    GenerateData = True

#------------------------------------------
# -------get numpy random state------------
np_state = np.random.get_state()

# --- basic settings ---
neurons_encoding = (origin_size[0] * origin_size[1]) / (pool_size[0] * pool_size[1])
reservoir_input = 1
blocks_input = 1
blocks_output = 1
blocks_reservoir = 10
neurons_block = 10

# --- parameters needs to be update by optimizer ---
gen = 'for example'

##-----------------------------------------------------
#--- create network ---
net = LSM_Network()
decoder = Decoder(config_group, config_key, config_SubCom, config_codes, config_ranges, config_borders,
                  config_precisions, config_scales)
generator = Generator(np_state)
generator.register_decoder(decoder)

generator.decoder.register(gen)

LSM_Network = generator.generate_network()
LSM_Network.join_netowrk(net)

generator.initialize(LSM_Network)

# ---------------------------------------
net.store('init')

inputs = zip(data_train_s, label_train)[0]
stimulus = TimedArray(inputs[0], dt=Dt)
duration = inputs[0].shape[0]
net.run(duration * Dt)




# --- parameters needs to be update by optimizer ---
# connect_matrix_block = np.array([[1,1,2,2],[3,5,4,6]])
# connect_matrix_blocks = [connect_matrix_block]*blocks_reservoir
# connect_matrix_reservoir = np.array([[1,2,3,4],[5,6,7,8]])
#
# tau_neurons = 30
#
# strength_synapse_block = np.random.rand((neurons_block*neurons_block))
# delay_synapse_block = np.random.rand((neurons_block*neurons_block)) * ms
#
# strength_synapse_reservoir = np.random.rand((blocks_reservoir*blocks_reservoir))
# delay_synapse_reservoir = np.random.rand((blocks_reservoir*blocks_reservoir)) * ms
#
# strength_synapse_encoding_reservoir = np.random.rand

##-----------------------------------------------------
#--- create network ---
# net = LSM_Network()
#
# #--- create encoding and readout neurons ---
# encoding = NeuronGroup(neurons_encoding, dynamics_encoding, threshold = 'I > 0', method = 'euler', refractory = 0 * ms,
#                         name = 'encoding')
# net.register_layer(encoding,'encoding')
#
# readout = NeuronGroup(neurons_block * blocks_reservoir, dynamics_readout, method = 'euler', name  = 'readout')
# net.register_layer(readout,'readout')
#
# #--- create reservoir ---
# reservoir = Reservoir(blocks_reservoir)
#
# # --- create blocks ---
# reservoir.create_blocks(neurons_block, connect_matrix_blocks, dynamics_reservoir, dynamics_synapse,
#                          dynamics_synapse_pre, threshold='v > 15', reset='v = 13.5', refractory=3 * ms)
#
# #--- create synapses between blocks in reservoir---
# reservoir.create_synapse(connect_matrix_reservoir, dynamics_synapse, dynamics_synapse_pre, blocks_input, blocks_output)
# reservoir.connect_blocks()
# net.register_layer(reservoir,'reservoir')
#
# #--- create synapses encoding and readout between reservoir---
# net.create_synapse_encoding(dynamics_synapse, dynamics_synapse_pre, reservoir_input)
# net.create_synapse_readout(dynamics_synapse_pre)
# net.connect_encoding()
# net.connect_readout()
# net.join_network()


# ------------------------------------
#--- initialize the parameters encoding and readout neurons ---
# encoding.property = '1'
# readout.v = '0'
# readout.g = '0'
# readout.tau = 30
#
# #--- initialize the parameters reservoir and blocks ---
# reservoir.initialize_blocks_neurons(v = np.array([13.5]*neurons_block) + np.random.rand(neurons_block)*1.5,
#                                     g = np.array([0] * neurons_block),
#                                     tau = np.array([tau_neurons] * neurons_block))
#
# reservoir.initialize_blocks_synapses(w = strength_synapse_block,
#                                      delay = delay_synapse_block)
#
# reservoir.initialize_reservoir_synapses(w = strength_synapse_reservoir,
#                                         delay = delay_synapse_reservoir)
#
# net.initialize_parameters_encoding_synapses(w = strength_synapse_encoding_reservoir)


