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
function = MathFunctions()
base = BaseFunctions()
readout = Readout()
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

# -------get numpy random state------------
# np_state = np.random.get_state()

neurons_encoding = (origin_size[0] * origin_size[1]) / (pool_size[0] * pool_size[1])
blocks_reservoir = 10
neurons_block = 10
ex_inh_ratio = 4
ex_neurons = int(neurons_block*(1/(ex_inh_ratio+1)))
inh_neurons = neurons_block - ex_neurons

connect_matrix = np.array([[1,1,2,2],[3,5,4,6]])

dynamics_encoding = '''
property : 1
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

#-----------------------------------------------------
net = Network()

encoding = NeuronGroup(neurons_encoding, dynamics_encoding, threshold='I > 0', method='euler', refractory=0 * ms,
                        name='encoding')
readout = NeuronGroup(neurons_block*blocks_reservoir, dynamics_readout, method='euler', name='neurongroup_read')

for index in range(blocks_reservoir):
    block_reservoir = Block(neurons_block)
    block_reservoir.create_neurons(dynamics_reservoir, threshold='v > 15', reset='v = 13.5', refractory=3 * ms,
                                   name='block_' + str(index))
    block_reservoir.create_synapse(dynamics_synapse, dynamics_synapse_pre, delay=1 * ms,
                                   name='block_block_' + str(index))
    block_reservoir.connect(connect_matrix)
    block_reservoir.join_networks(net)

    synapse_encoding_reservoir = Synapses(encoding, block_reservoir.neurons, dynamics_synapse,
                                          on_pre=dynamics_synapse_pre,
                                          method='euler', name='encoding_block_' + str(index))
    synapse_encoding_reservoir.connect(j='i')
    net.add(synapse_encoding_reservoir)

    synapse_reservoir_readout = Synapses(block_reservoir.neurons, readout, 'w = 1 : 1',
                                         on_pre=dynamics_synapse_pre, name='readout_' + str(index))

    synapse_reservoir_readout.connect(j='i+' + str(index * neurons_block))
    net.add(synapse_reservoir_readout)

    # -------initialization of neuron parameters----------
    block_reservoir.neurons.property = np.array(([-1] * ex_neurons) + ([1] * inh_neurons))
    block_reservoir.neurons.v = '13.5+1.5*rand()'
    block_reservoir.neurons.g = '0'
    block_reservoir.neurons.tau = 30

    block_reservoir.synapse.w = 'rand()'
    synapse_encoding_reservoir.w = 'rand()'

encoding.property = '1'
readout.g = '0'
readout.g = '0'
readout.tau = 30

net.add(encoding, readout)
#net.store('init')

inputs = zip(data_train_s, label_train)[0]
stimulus = TimedArray(inputs[0], dt=Dt)
duration = inputs[0].shape[0]
net.run(duration * Dt)