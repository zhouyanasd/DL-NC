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
Add the project path to the PYTHONPATH or site-packages for linux system.

Citation
=======

"""
import os, sys
exec_dir = os.path.split(os.path.realpath(__file__))[0]
project_dir = os.path.split(os.path.split(os.path.split(exec_dir)[0])[0])[0]

sys.path.append(project_dir)
data_path = '/home/zy/Project/DL-NC'+'/Data/KTH/'
LHS_path = exec_dir+'/LHS_KTH.dat'

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.ray_config import *

from brian2 import *
from sklearn.preprocessing import MinMaxScaler

import gc
from functools import partial

import ray
from ray.util.multiprocessing import Pool
from ray.exceptions import RayActorError, WorkerCrashedError, RayError

ray_cluster_address = 'auto'

exec_env = '''
from brian2 import *
import warnings
try:
    clear_cache('cython')
except:
    pass
warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
'''

exec_var = open(os.path.join(exec_dir,"src/config.py")).read()

# -------prepare the ray cluster------------
cluster = Cluster(ray_cluster_one)
if ray.is_initialized():
    ray.shutdown()
try:
    cluster.reconnect(cluster.check_alive())
    ray.init(address=ray_cluster_address, logging_level=logging.ERROR)
except ConnectionError:
    cluster.start()

###################################
#------------------------------------------
# -------get numpy random state------------
np.random.seed(100)
np_state = np.random.get_state()
ob_np_state = ray.put(np_state)

# -----simulation parameter setting-------
core = 60

method = 'CoE_rf'
total_eva = 300
load_continue = False

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

data_train_index_batch = KTH.data_batch(df_en_train.index.values, core)
data_pre_train_index_batch = KTH.data_batch(df_en_pre_train.index.values, core)
data_validation_index_batch = KTH.data_batch(df_en_validation.index.values, core)
data_test_index_batch = KTH.data_batch(df_en_test.index.values, core)

#--- define network run function ---
def init_net(gen):
    exec(exec_env)
    exec(exec_var)

    # ---- set numpy random state for each run----
    np.random.set_state(ray.get(ob_np_state))

    # ---- register the gen to the decoder ----
    generator.decoder.register(gen)

    #--- create network ---
    net = Network()
    LSM_network = generator.generate_network()
    generator.initialize(LSM_network)
    generator.join(net, LSM_network)
    net.store('init')
    return net

def pre_run_net(gen, data_index):
    exec(exec_env)
    exec(exec_var)
    KTH_ = KTH_classification()
    df_en_pre_train = KTH_.load_data(data_path + 'pre_train_' + DataName + '.p')
    data_pre_train_s, label_pre_train = KTH_.get_series_data_list(df_en_pre_train, is_group=True)

    #--- run network ---
    net = init_net(gen)
    Switch = 1
    for ind in data_index:
        data = data_pre_train_s[ind]
        stimulus = TimedArray(data, dt=Dt)
        duration = data.shape[0]
        net.run(duration * Dt)
    return net._full_state()

def sum_strength(gen, net_state_list):
    net = init_net(gen)
    state_init = net._full_state()
    l = len(net_state_list)
    for com in list(state_init.keys()):
        if 'block_block_' in com or 'pathway_' in com and '_pre' not in com and '_post' not in com:
            try:
                np.subtract(state_init[com]['strength'][0], state_init[com]['strength'][0],
                       out=state_init[com]['strength'][0])
                for state in net_state_list:
                    np.add(state_init[com]['strength'][0], state[com]['strength'][0]/l,
                            out = state_init[com]['strength'][0])
            except:
                continue
    return state_init

def run_net(gen, state_pre_run, data_indexs):
    exec(exec_env)
    exec(exec_var)
    KTH_ = KTH_classification()
    df_en_train = KTH_.load_data(data_path + 'train_' + DataName+'.p')
    df_en_validation = KTH_.load_data(data_path + 'validation_' + DataName+'.p')
    df_en_test = KTH_.load_data(data_path + 'test_' + DataName+'.p')
    data_train_s, label_train = KTH_.get_series_data_list(df_en_train, is_group=True)
    data_validation_s, label_validation = KTH_.get_series_data_list(df_en_validation, is_group=True)
    data_test_s, label_test = KTH_.get_series_data_list(df_en_test, is_group=True)

    #--- run network ---
    net = init_net(gen)
    net._stored_state['pre_run'] = state_pre_run
    net.restore('pre_run')
    Switch = 0
    states_train = []
    labels_train = []
    states_val = []
    labels_val = []
    states_test = []
    labels_test = []

    for ind in data_indexs[0]:
        data = data_train_s[ind]
        stimulus = TimedArray(data, dt=Dt)
        duration = data.shape[0]
        net.run(duration * Dt)
        state = net.get_states()['block_readout']['v']
        states_train.append(state)
        labels_train.append(label_train[ind])
        net.restore('pre_run')

    for ind in data_indexs[1]:
        data = data_validation_s[ind]
        stimulus = TimedArray(data, dt=Dt)
        duration = data.shape[0]
        net.run(duration * Dt)
        state = net.get_states()['block_readout']['v']
        states_val.append(state)
        labels_val.append(label_validation[ind])
        net.restore('pre_run')

    for ind in data_indexs[2]:
        data = data_test_s[ind]
        stimulus = TimedArray(data, dt=Dt)
        duration = data.shape[0]
        net.run(duration * Dt)
        state = net.get_states()['block_readout']['v']
        states_test.append(state)
        labels_test.append(label_test[ind])
        net.restore('pre_run')

    return states_train, labels_train, states_val, labels_val, states_test, labels_test

def parallel_run(fun, data):
    while True:
        try:
            # ------apply the pool-------
            pool = Pool(processes=core, ray_address=ray_cluster_address, maxtasksperchild=None)
            result = pool.map(fun, [x for x in data])
            # ------close the pool-------
            pool.close()
            pool.join()
            return result
        except (RayActorError, WorkerCrashedError) as e:
            print(repr(e))
            cluster.reconnect(cluster.check_alive())
        except RayError as e:
            print(repr(e))
            cluster.restart()
        except Exception as e:
            print(repr(e))
        finally:
            print('unknown error, restart task')

@ProgressBar
@Timelog
def parameters_search(**parameter):
    # ------convert the parameter to gen-------
    gen = [parameter[key] for key in decoder.get_keys]
    # ------init net and run for pre_train-------
    net_state_list = parallel_run(partial(pre_run_net, gen), data_pre_train_index_batch)
    state_pre_run = sum_strength(gen, net_state_list)
    # ------parallel run for training data-------
    results_list = parallel_run(partial(run_net, gen, state_pre_run), zip(data_train_index_batch,
                                                                          data_validation_index_batch,
                                                                          data_test_index_batch))
    # ------Readout---------------
    states_train, states_validation, states_test, _label_train, _label_validation, _label_test = [], [], [], [], [], []
    for result in results_list:
        states_train.extend(result[0])
        _label_train.extend(result[1])
        states_validation.extend(result[2])
        _label_validation.extend(result[3])
        states_test.extend(result[4])
        _label_test.extend(result[5])
    states_train = (MinMaxScaler().fit_transform(np.asarray(states_train))).T
    states_validation = (MinMaxScaler().fit_transform(np.asarray(states_validation))).T
    states_test = (MinMaxScaler().fit_transform(np.asarray(states_test))).T
    score_train, score_validation, score_test = evaluator.readout_sk(states_train, states_validation, states_test,
                                                                   np.asarray(_label_train),
                                                                   np.asarray(_label_validation),
                                                                   np.asarray(_label_test), solver="lbfgs",
                                                                   multi_class="multinomial")
    # ------collect the memory-------
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
    parameters_search.total = total_eva
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
                                                      init_points=150,
                                                      maxormin=1, SUBPOP=1, GGAP=0.5, online=False, eva=1,
                                                      interval=1,
                                                      selectStyle='sus', recombinStyle='xovdp',
                                                      distribute=False, LHS_path = LHS_path, drawing=False,
                                                      load_continue = load_continue)

    elif method == 'CoE_gp':
        optimizer = CoE_surrogate_mixgentype(parameters_search, None, decoder.get_SubCom, decoder.get_ranges,
                                             decoder.get_borders, decoder.get_precisions, decoder.get_codes,
                                             decoder.get_scales, decoder.get_keys, None,
                                             surrogate_type='gp', acq='ucb', kappa=2.576, xi=0.0)
        best_gen, best_ObjV = optimizer.coe_surrogate(recopt=0.9, pm=0.1, MAXGEN=50, NIND=10,
                                                      init_points=150,
                                                      maxormin=1, SUBPOP=1, GGAP=0.5, online=False, eva=1,
                                                      interval=1,
                                                      selectStyle='sus', recombinStyle='xovdp',
                                                      distribute=False, LHS_path = LHS_path, drawing=True,
                                                      load_continue = load_continue)

    cluster.stop()