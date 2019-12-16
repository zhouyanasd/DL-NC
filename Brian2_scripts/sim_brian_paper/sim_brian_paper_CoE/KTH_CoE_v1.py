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

n_input = (origin_size[0] * origin_size[1]) / (pool_size[0] * pool_size[1])

dynamics_spike_train = '''
I = stimulus(t,i) : 1
'''

dynamics_neuron = '''
property : 1
tau : 1
dv/dt = (I-v) / (tau*ms) : 1 (unless refractory)
dg/dt = (-g)/(3*ms) : 1
I = g+13.5 : 1
'''

# neuron_read = '''
# tau : 1
# dv/dt = (I-v) / (tau*ms) : 1
# dg/dt = (-g)/(3*ms) : 1
# dh/dt = (-h)/(6*ms) : 1
# I = (g+h): 1
# '''

dynamics_synapse = '''
w : 1
'''

dynamics_pre_synapse = '''
g += w * property
'''

# on_pre_inh = '''
# h-=w
# '''

connect_matrix = np.array([[1,1,2,2],[3,3,4,4]])

net = Network(collect())

block_input = Block()

block = Block(10)
block.create_neurons(dynamics_neuron, threshold='v > 15', reset='v = 13.5',  refractory=3 * ms)
block.create_synapse(dynamics_synapse, dynamics_pre_synapse)
block.connect(connect_matrix)
block.join_networks(net)


