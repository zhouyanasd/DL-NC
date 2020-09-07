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

from functools import partial
from multiprocessing import Pool

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
# math_function = MathFunctions()
# base_function = BaseFunctions()
# evaluate = Evaluation()
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

# # --- parameters needs to be update by optimizer ---
# gen = 'for example'

##-----------------------------------------------------
#--- create generator and decoder ---
decoder = Decoder(config_group, config_keys, config_SubCom, config_codes, config_ranges, config_borders,
                  config_precisions, config_scales)
generator = Generator(np_state)
generator.register_decoder(decoder)

#--- define network run function ---
def run_net(inputs, gen):
    generator.decoder.register(gen)

    #--- create network ---
    net = Network()
    LSM_network = generator.generate_and_initialize()
    LSM_network.join_network(net)
    net.store('init')

    #--- run network ---
    inputs = zip(data_train_s, label_train)[0]
    stimulus = TimedArray(inputs[0], dt=Dt)
    duration = inputs[0].shape[0]
    net.run(duration * Dt)
    states = net.get_states()['neurongroup_read']['v']
    net.restore('init')
    return (states, inputs[1])

@Timelog
@AddParaName
def parameters_search(**parameter):
    pass

##########################################
# -------optimizer settings---------------
if __name__ == '__main__':
    core = 8
    pool = Pool(core)

    method = 'CoE'
    surrogate = 'rf'

# -------parameters search---------------
    if method == 'BO':
        optimizer = BayesianOptimization(
            f=parameters_search,
            # pbounds=bounds,
            random_state=np.random.RandomState(),
        )

    elif method == 'CoE':
        optimizer = Coe_surrogate_mixgentype()



# delay_synapse_block = np.random.rand((neurons_block*neurons_block)) * ms


