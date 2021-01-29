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
from sklearn.preprocessing import MinMaxScaler

import os, gc, warnings
from functools import partial
from multiprocessing import Manager, Pool

warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)
if os.name == 'nt':
    data_path = '../../../Data/KTH/'
elif os.name == 'posix':
    os.system('export PYTHONPATH=~/Project/DL-NC')
    data_path = '/home/zy/Project/DL-NC/Data/KTH/'

###################################
#------------------------------------------
# -------get numpy random state------------
np_state = np.random.get_state()

# -----simulation parameter setting-------
DataName = 'coe_[15,5,4]'

origin_size = (120, 160)
pool_size = (5, 5)
pool_types = 'max'
pool_threshold = 0.2

F_train = 1
F_pre_train = 0.2
F_validation = 1
F_test = 1

neurons_encoding = int((origin_size[0] * origin_size[1]) / (pool_size[0] * pool_size[1]))

##-----------------------------------------------------
# --- data and classifier ----------------------
KTH = KTH_classification()
evaluator = Evaluation()

#--- create generator and decoder ---
decoder = Decoder(config_group, config_keys, config_SubCom, config_codes, config_ranges, config_borders,
                  config_precisions, config_scales, neurons_encoding, gen_group)

generator = Generator(np_state)
generator.register_decoder(decoder)

# -------data initialization----------------------
try:
    df_en_train = KTH.load_data(data_path + 'train_' + DataName+'.p')
    df_en_pre_train = KTH.load_data(data_path + 'pre_train_' + DataName + '.p')
    df_en_validation = KTH.load_data(data_path + 'validation_' + DataName+'.p')
    df_en_test = KTH.load_data(data_path + 'test_' + DataName+'.p')
except FileNotFoundError:
    KTH.load_data_KTH_all(data_path, split_type='mixed', split=[15, 5, 4])

    df_train = KTH.select_data_KTH(F_train, KTH.train, False)
    df_pre_train = KTH.select_data_KTH(F_pre_train, KTH.train, False)
    df_validation = KTH.select_data_KTH(F_validation, KTH.validation, False)
    df_test = KTH.select_data_KTH(F_train, KTH.test, False)

    df_en_train = KTH.encoding_latency_KTH(df_train, origin_size, pool_size, pool_types, pool_threshold)
    df_en_pre_train = KTH.encoding_latency_KTH(df_pre_train, origin_size, pool_size, pool_types, pool_threshold)
    df_en_validation = KTH.encoding_latency_KTH(df_validation, origin_size, pool_size, pool_types, pool_threshold)
    df_en_test = KTH.encoding_latency_KTH(df_test, origin_size, pool_size, pool_types, pool_threshold)

    KTH.dump_data(data_path + 'train_' + DataName + '.p', df_en_train)
    KTH.dump_data(data_path + 'pre_train_' + DataName + '.p', df_en_pre_train)
    KTH.dump_data(data_path + 'validation_' + DataName + '.p', df_en_validation)
    KTH.dump_data(data_path + 'test_' + DataName + '.p', df_en_test)

data_train_s, label_train = KTH.get_series_data_list(df_en_train, is_group=True)
data_pre_train_s, label_pre_train = KTH.get_series_data_list(df_en_pre_train, is_group=True)
data_validation_s, label_validation = KTH.get_series_data_list(df_en_validation, is_group=True)
data_test_s, label_test = KTH.get_series_data_list(df_en_test, is_group=True)

#--- define network run function ---
def init_net(gen):
    # ---- set numpy random state for each run----
    np.random.set_state(np_state)

    # ---- register the gen to the decoder ----
    generator.decoder.register(gen)

    #--- create network ---
    net = Network()
    LSM_network = generator.generate_network()
    generator.initialize(LSM_network)
    generator.join(net, LSM_network)
    net.store('init')
    return net

def pre_run_net(gen, inputs, queue):
    #--- run network ---
    global Switch, stimulus
    net = init_net(gen)
    Switch = 1
    state = queue.get(False)
    net._stored_state['temp'] = state
    net.restore('temp')
    stimulus = TimedArray(inputs[0], dt=Dt)
    duration = inputs[0].shape[0]
    net.run(duration * Dt)
    queue.put(net._full_state(), False)

def sum_strength(gen, queue):
    net = init_net(gen)
    state_init = net._full_state()
    l = queue.qsize()
    states = []
    while not queue.empty():
        states.append(queue.get())
    for com in list(state_init.keys()):
        if 'block_block_' in com or 'pathway_' in com and '_pre' not in com and '_post' not in com:
            try:
                np.subtract(state_init[com]['strength'][0], state_init[com]['strength'][0],
                       out=state_init[com]['strength'][0])
                for state in states:
                    np.add(state_init[com]['strength'][0], state[com]['strength'][0]/l,
                            out = state_init[com]['strength'][0])
            except:
                continue
    net._stored_state['pre_run'] = state_init
    net.store('pre_run', 'pre_run_state.txt')

def run_net(gen, inputs):
    #--- run network ---
    global Switch, stimulus
    net = init_net(gen)
    net.restore('pre_run', 'pre_run_state.txt')
    Switch = 0
    stimulus = TimedArray(inputs[0], dt=Dt)
    duration = inputs[0].shape[0]
    net.run(duration * Dt)
    states = net.get_states()['block_readout']['v']
    return (states, inputs[1])

@ProgressBar
@Timelog
def parameters_search(**parameter):
    # ------apply the pool and queue-------
    pool = Pool(core)
    q = Manager().Queue(core)
    # ------convert the parameter to gen-------
    gen = [parameter[key] for key in decoder.get_keys]
    # ------init net and run for pre_train-------
    net = init_net(gen)
    for i in range(core):
        q.put(net._full_state())
    pool.starmap(partial(pre_run_net, gen), [(x, q) for x in zip(data_pre_train_s, label_pre_train)])
    sum_strength(gen, q)
    # ------parallel run for train-------
    states_train_list = pool.map(partial(run_net, gen), [(x) for x in zip(data_train_s, label_train)])
    # ------parallel run for validation-------
    states_validation_list = pool.map(partial(run_net, gen), [(x) for x in zip(data_validation_s, label_validation)])
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
    # ------close the pool and collect the memory-------
    pool.close()
    pool.join()
    del net, q, pool
    gc.collect()
    # ----------show results-----------
    print('parameter %s' % parameter)
    print('Train score: ', score_train)
    print('Validation score: ', score_validation)
    print('Test score: ', score_test)
    return 1 - score_validation, 1 - score_test, 1 - score_train, parameter

##########################################
# -------optimizer settings---------------
if __name__ == '__main__':
    core = 1
    parameters_search.total = 600

    method = 'CoE_rf'
    LHS_path = './LHS_KTH.dat'
    load_continue = False
    parameters_search.load_continue = load_continue
    parameters_search.func.load_continue = load_continue

# -------parameters search---------------
    if method == 'BO':
        optimizer = BayesianOptimization(
            f=parameters_search,
            keys = decoder.get_keys,
            ranges = decoder.get_ranges,
            borders = decoder.get_borders,
            precisions = decoder.get_precisions,
            random_state=np.random.RandomState(),
        )
        optimizer.minimize(
            LHS_path=LHS_path,
            init_points=300,
            is_LHS=True,
            n_iter=300,
        )

    elif method == 'CoE':
        optimizer = CoE_surrogate_mixgentype(parameters_search, None, decoder.get_SubCom, decoder.get_ranges,
                                             decoder.get_borders, decoder.get_precisions, decoder.get_codes,
                                             decoder.get_scales, decoder.get_keys, None,
                                             )
        best_gen, best_ObjV = optimizer.coe(recopt=0.9, pm=0.1, MAXGEN=9, NIND=10,
                                            maxormin=1, SUBPOP=1, GGAP=0.5,
                                            selectStyle='sus', recombinStyle='xovdp',
                                            distribute=False, drawing=False, load_continue = load_continue)

    elif method == 'CoE_rf':
        optimizer = CoE_surrogate_mixgentype(parameters_search, None, decoder.get_SubCom, decoder.get_ranges,
                                             decoder.get_borders, decoder.get_precisions, decoder.get_codes,
                                             decoder.get_scales, decoder.get_keys, None,
                                             surrogate_type = 'rf', n_Q = 100, n_estimators=1000)
        best_gen, best_ObjV = optimizer.coe_surrogate(recopt=0.9, pm=0.1, MAXGEN=50, NIND=10,
                                                      init_points=300,
                                                      maxormin=1, SUBPOP=1, GGAP=0.5, online=False, eva=2,
                                                      interval=2,
                                                      selectStyle='sus', recombinStyle='xovdp',
                                                      distribute=False, LHS_path = LHS_path, drawing=False,
                                                      load_continue = load_continue)

    elif method == 'CoE_gp':
        optimizer = CoE_surrogate_mixgentype(parameters_search, None, decoder.get_SubCom, decoder.get_ranges,
                                             decoder.get_borders, decoder.get_precisions, decoder.get_codes,
                                             decoder.get_scales, decoder.get_keys, None,
                                             surrogate_type='gp', acq='ucb', kappa=2.576, xi=0.0)
        best_gen, best_ObjV = optimizer.coe_surrogate(recopt=0.9, pm=0.1, MAXGEN=50, NIND=10,
                                                      init_points=300,
                                                      maxormin=1, SUBPOP=1, GGAP=0.5, online=False, eva=2,
                                                      interval=2,
                                                      selectStyle='sus', recombinStyle='xovdp',
                                                      distribute=False, LHS_path = LHS_path, drawing=True,
                                                      load_continue = load_continue)



