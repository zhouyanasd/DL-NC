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
from sklearn.preprocessing import MinMaxScaler
import geatpy as ga

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
KTH = KTH_classification()
evaluator = Evaluation()

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

##-----------------------------------------------------
#--- create generator and decoder ---
decoder = Decoder(config_group, config_keys, config_SubCom, config_codes, config_ranges, config_borders,
                  config_precisions, config_scales, neurons_encoding)

generator = Generator(np_state)
generator.register_decoder(decoder)

#--- define network run function ---
def run_net(inputs, gen):
    generator.decoder.register(gen)

    #--- create network ---
    net = Network()
    LSM_network = generator.generate_network()
    generator.initialize(LSM_network)
    generator.join(net, LSM_network)
    net.store('init')

    #--- run network ---
    # inputs = zip(data_train_s, label_train)[0]
    stimulus = TimedArray(inputs[0], dt=Dt)
    duration = inputs[0].shape[0]
    net.run(duration * Dt)
    states = net.get_states()['block_readout']['v']
    net.restore('init')
    return (states, inputs[1])

@Timelog
@AddParaName
def parameters_search(gen):
    # ------parallel run for train-------
    states_train_list = pool.map(partial(run_net, gen), [(x) for x in zip(data_train_s, label_train)])
    # ------parallel run for validation-------
    states_validation_list = pool.map(partial(run_net, gen),
                                      [(x) for x in zip(data_validation_s, label_validation)])
    # ----parallel run for test--------
    states_test_list = pool.map(partial(run_net, gen), [(x) for x in zip(data_test_s, label_test)])
    # ------Readout---------------
    states_train, states_validation, states_test, _label_train, _label_validation, _label_test = [], [], [], [], [], []
    for train in states_train_list:
        states_train.append(train[0])
        _label_train.append(train[1])
    for validation in states_validation_list:
        states_validation.append(validation[0])
        _label_validation.append(validation[1])
    for test in states_test_list:
        states_test.append(test[0])
        _label_test.append(test[1])
    states_train = (MinMaxScaler().fit_transform(np.asarray(states_train))).T
    states_validation = (MinMaxScaler().fit_transform(np.asarray(states_validation))).T
    states_test = (MinMaxScaler().fit_transform(np.asarray(states_test))).T
    score_train, score_validation, score_test = evaluator.readout_sk(states_train, states_validation, states_test,
                                                                   np.asarray(_label_train),
                                                                   np.asarray(_label_validation),
                                                                   np.asarray(_label_test), solver="lbfgs",
                                                                   multi_class="multinomial")
    # ----------show results-----------
    print('gen %s' % gen)
    print('Train score: ', score_train)
    print('Validation score: ', score_validation)
    print('Test score: ', score_test)
    return 1 - score_validation, 1 - score_test, 1 - score_train, gen

##########################################
# -------optimizer settings---------------
if __name__ == '__main__':
    core = 8
    pool = Pool(core)

    method = 'CoE_rf'
    LHS_path = './LHS_KTH.dat'

# -------parameters search---------------
    if method == 'BO':
        optimizer = BayesianOptimization(
            f=parameters_search,
            pbounds=dict(zip(config_keys, [tuple(x) for x in
                                           ga.crtfld(config_ranges, config_borders, list(config_precisions)).T])),
            random_state=np.random.RandomState(),
        )

    elif method == 'CoE_rf':
        optimizer = Coe_surrogate_mixgentype(parameters_search, None, config_SubCom, config_ranges, config_borders,
                                             config_precisions, config_codes, config_scales, config_keys, np_state,
                                             surrogate_type = 'rf', n_Q = 100, n_estimators=1000)
        best_gen, best_ObjV = optimizer.coe_surrogate_real_templet(recopt=0.9, pm=0.1, MAXGEN=100, NIND=10,
                                                                   init_points=50, problem='R',
                                                                   maxormin=1, SUBPOP=1, GGAP=0.5, online=False, eva=1,
                                                                   interval=1,
                                                                   selectStyle='sus', recombinStyle='xovdp',
                                                                   distribute=False, drawing=False)

    elif method == 'CoE_gp':
        optimizer = Coe_surrogate_mixgentype(parameters_search, None, config_SubCom, config_ranges, config_borders,
                                             config_precisions, config_codes, config_scales, config_keys, np_state,
                                             surrogate_type='gp', acq='ucb', kappa=2.576, xi=0.0)
        best_gen, best_ObjV = optimizer.coe_surrogate_real_templet(recopt=0.9, pm=0.1, MAXGEN=100, NIND=10,
                                                                   init_points=50, problem='R',
                                                                   maxormin=1, SUBPOP=1, GGAP=0.5, online=False, eva=1,
                                                                   interval=1,
                                                                   selectStyle='sus', recombinStyle='xovdp',
                                                                   distribute=False, drawing=True)



