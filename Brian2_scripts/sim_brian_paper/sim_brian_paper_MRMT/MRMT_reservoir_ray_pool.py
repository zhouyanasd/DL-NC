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
project_dir_sever = '/home/zy/Project/DL-NC'
exec_dir_sever = exec_dir.replace(project_dir, project_dir_sever)
sys.path.append(project_dir)

data_path_HAPT = project_dir_sever + '/Data/HAPT-Dataset/'
data_path_KTH = project_dir_sever+'/Data/KTH/'
data_path_NMNIST = project_dir_sever + '/Data/N_MNIST/'

LHS_path_HAPT = exec_dir + '/LHS_HAPT.dat'
LHS_path_KTH = exec_dir +'/LHS_KTH.dat'
LHS_path_NMNIST = exec_dir + '/LHS_NMNIST.dat'

Optimal_gens = exec_dir + '/Optimal_gens.pkl'

from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src.config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src.ray_config import *

from brian2 import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
BrianLogger.log_level_error()
prefs.codegen.target = "numpy"
start_scope()
'''

exec_var = open(os.path.join(exec_dir, "src/config.py")).read()

# -------prepare the ray cluster------------
cluster = Cluster(ray_cluster_one)
cluster.sync_file_cluster(exec_dir, exec_dir_sever, '.py')
if ray.is_initialized():
    ray.shutdown()
try:
    cluster.reconnect(cluster.check_alive())
except ConnectionError:
    cluster.start()
ray.init(address=ray_cluster_address, logging_level=logging.ERROR)

###################################
# ------------------------------------------
# -------get numpy random state------------
seed = 100
np.random.seed(seed)
np_state = np.random.get_state()

# -----simulation overall setting-------
core = 60

method = 'GA'
total_eva = 600
load_continue = False
Training_task = 'KTH'

# -----simulation data setting-------
DataName_HAPT = 'coe_0.5'
DataName_KTH = 'coe_[15,5,4]'
DataName_NMNIST = 'coe_0.05'

F_train_HAPT = 0.5
F_pre_train_HAPT = 0.02
F_validation_HAPT = 0.3
F_test_HAPT = 0.5

F_train_KTH = 1
F_pre_train_KTH = 0.2
F_validation_KTH = 1
F_test_KTH = 1

F_train_NMNIST = 0.05833333
F_pre_train_NMNIST = 0.002
F_validation_NMNIST = 0.1666667
F_test_NMNIST = 0.05

# -----simulation encoding setting-------
coding_duration_HAPT = 30
coding_n_HAPT = 3
neurons_encoding_HAPT = 561 * coding_n_HAPT

origin_size_KTH = (120, 160)
pool_size_KTH = (5, 5)
pool_types_KTH = 'max'
pool_threshold_KTH = 0.2
neurons_encoding_KTH = int((origin_size_KTH[0] * origin_size_KTH[1]) / (pool_size_KTH[0] * pool_size_KTH[1]))

coding_duration_NMNIST = 10
shape_NMNIST = (34, 34)
neurons_encoding_NMNIST = 2 * shape[0] * shape[1]

##-----------------------------------------------------
# --- data and classifier ----------------------
HAPT = HAPT_classification(coding_duration_HAPT)
KTH = KTH_classification()
NMNIST = NMNIST_classification(shape, neurons_encoding_NMNIST)

evaluator = Evaluation()

# --- create generator and decoder based on task---
decoder = Decoder_Reservoir(config_group_reservoir, config_keys_reservoir, config_SubCom_reservoir,
                            config_codes_reservoir, config_ranges_reservoir, config_borders_reservoir,
                            config_precisions_reservoir, config_scales_reservoir, gen_group_reservoir)
generator = Generator_Reservoir(np_state)
generator.register_decoder(decoder, block_max=block_max)
optimal_block_gens = decoder.load_data(Optimal_gens)
decoder.register_optimal_block_gens(optimal_block_gens)
generator.register_block_generator(HAPT=neurons_encoding_HAPT, KTH=neurons_encoding_KTH, NMNIST= neurons_encoding_NMNIST)
generator.initialize_task_ids()

# -------data initialization for all tasks----------------------
try:
    df_en_train_HAPT = HAPT.load_data(data_path_HAPT + 'Spike_train_Data/train_' + DataName_HAPT + '.p')
    df_en_pre_train_HAPT = HAPT.load_data(data_path_HAPT + 'Spike_train_Data/pre_train_' + DataName_HAPT + '.p')
    df_en_validation_HAPT = HAPT.load_data(data_path_HAPT + 'Spike_train_Data/validation_' + DataName_HAPT + '.p')
    df_en_test_HAPT = HAPT.load_data(data_path_HAPT + 'Spike_train_Data/test_' + DataName_HAPT + '.p')

    df_en_train_KTH = KTH.load_data(data_path_KTH + 'train_' + DataName_KTH +'.p')
    df_en_pre_train_KTH = KTH.load_data(data_path_KTH + 'pre_train_' + DataName_KTH + '.p')
    df_en_validation_KTH = KTH.load_data(data_path_KTH + 'validation_' + DataName_KTH +'.p')
    df_en_test_KTH = KTH.load_data(data_path_KTH + 'test_' + DataName_KTH +'.p')

    df_en_train_NMNIST = NMNIST.load_data(data_path_NMNIST + 'train_' + DataName_NMNIST +'.p')
    df_en_pre_train_NMNIST = NMNIST.load_data(data_path_NMNIST + 'pre_train_' + DataName_NMNIST + '.p')
    df_en_validation_NMNIST = NMNIST.load_data(data_path_NMNIST + 'validation_' + DataName_NMNIST +'.p')
    df_en_test_NMNIST = NMNIST.load_data(data_path_NMNIST + 'test_' + DataName_NMNIST +'.p')
except FileNotFoundError:
    HAPT.load_data_UCI_all(data_path_HAPT)
    KTH.load_data_KTH_all(data_path_KTH, split_type='mixed', split=[15, 5, 4])
    NMNIST.load_data_NMNIST_all(data_path_NMNIST)

    df_train_validation_HAPT = HAPT.select_data(F_train_HAPT, HAPT.train, False, selected=[1, 2, 3, 4, 5, 6])
    df_pre_train_HAPT = HAPT.select_data(F_pre_train_HAPT, HAPT.train, False, selected=[1, 2, 3, 4, 5, 6])
    df_train_HAPT, df_validation_HAPT = train_test_split(df_train_validation_HAPT, test_size=F_validation_HAPT, random_state=42)
    df_test_HAPT = HAPT.select_data(F_test_HAPT, HAPT.test, False, selected=[1, 2, 3, 4, 5, 6])

    df_train_KTH = KTH.select_data_KTH(F_train_KTH, KTH.train, False)
    df_pre_train_KTH = KTH.select_data_KTH(F_pre_train_KTH, KTH.train, False)
    df_validation_KTH = KTH.select_data_KTH(F_validation_KTH, KTH.validation, False)
    df_test_KTH = KTH.select_data_KTH(F_train_KTH, KTH.test, False)

    df_train_validation_NMNIST = NMNIST.select_data_NMNIST(F_train_NMNIST, NMNIST.train, False)
    df_pre_train_NMNIST = NMNIST.select_data_NMNIST(F_pre_train_NMNIST, NMNIST.train, False)
    df_train_NMNIST, df_validation_NMNIST = train_test_split(df_train_validation_NMNIST, test_size=F_validation_NMNIST, random_state=42)
    df_test_NMNIST = NMNIST.select_data_NMNIST(F_test_NMNIST, NMNIST.test, False)

    df_en_train_HAPT = HAPT.encoding_latency_UCI(HAPT._encoding_cos_rank_ignore_0, df_train_HAPT, coding_n_HAPT)
    df_en_pre_train_HAPT = HAPT.encoding_latency_UCI(HAPT._encoding_cos_rank_ignore_0, df_pre_train_HAPT, coding_n_HAPT)
    df_en_validation_HAPT = HAPT.encoding_latency_UCI(HAPT._encoding_cos_rank_ignore_0, df_validation_HAPT, coding_n_HAPT)
    df_en_test_HAPT = HAPT.encoding_latency_UCI(HAPT._encoding_cos_rank_ignore_0, df_test_HAPT, coding_n_HAPT)

    df_en_train_KTH = KTH.encoding_latency_KTH(df_train_KTH, origin_size_KTH, pool_size_KTH, pool_types_KTH, pool_threshold_KTH)
    df_en_pre_train_KTH = KTH.encoding_latency_KTH(df_pre_train_KTH, origin_size_KTH, pool_size_KTH, pool_types_KTH, pool_threshold_KTH)
    df_en_validation_KTH = KTH.encoding_latency_KTH(df_validation_KTH, origin_size_KTH, pool_size_KTH, pool_types_KTH, pool_threshold_KTH)
    df_en_test_KTH = KTH.encoding_latency_KTH(df_test_KTH, origin_size_KTH, pool_size_KTH, pool_types_KTH, pool_threshold_KTH)

    df_en_train_NMNIST = NMNIST.encoding_latency_NMNIST(df_train_NMNIST)
    df_en_pre_train_NMNIST = NMNIST.encoding_latency_NMNIST(df_pre_train_NMNIST)
    df_en_validation_NMNIST = NMNIST.encoding_latency_NMNIST(df_validation_NMNIST)
    df_en_test_NMNIST = NMNIST.encoding_latency_NMNIST(df_test_NMNIST)

    HAPT.dump_data(data_path_HAPT + 'Spike_train_Data/train_' + DataName_HAPT + '.p', df_en_train_HAPT)
    HAPT.dump_data(data_path_HAPT + 'Spike_train_Data/pre_train_' + DataName_HAPT + '.p', df_en_pre_train_HAPT)
    HAPT.dump_data(data_path_HAPT + 'Spike_train_Data/validation_' + DataName_HAPT + '.p', df_en_validation_HAPT)
    HAPT.dump_data(data_path_HAPT + 'Spike_train_Data/test_' + DataName_HAPT + '.p', df_en_test_HAPT)

    KTH.dump_data(data_path_KTH + 'train_' + DataName_KTH + '.p', df_en_train_KTH)
    KTH.dump_data(data_path_KTH + 'pre_train_' + DataName_KTH + '.p', df_en_pre_train_KTH)
    KTH.dump_data(data_path_KTH + 'validation_' + DataName_KTH + '.p', df_en_validation_KTH)
    KTH.dump_data(data_path_KTH + 'test_' + DataName_KTH + '.p', df_en_test_KTH)

    NMNIST.dump_data(data_path_NMNIST + 'train_' + DataName_NMNIST + '.p', df_en_train_NMNIST)
    NMNIST.dump_data(data_path_NMNIST + 'pre_train_' + DataName_NMNIST + '.p', df_en_pre_train_NMNIST)
    NMNIST.dump_data(data_path_NMNIST + 'validation_' + DataName_NMNIST + '.p', df_en_validation_NMNIST)
    NMNIST.dump_data(data_path_NMNIST + 'test_' + DataName_NMNIST + '.p', df_en_test_NMNIST)

data_train_index_batch_HAPT = HAPT.data_batch(df_en_train_HAPT.index.values, core)
data_pre_train_index_batch_HAPT = HAPT.data_batch(df_en_pre_train_HAPT.index.values, core)
data_validation_index_batch_HAPT = HAPT.data_batch(df_en_validation_HAPT.index.values, core)
data_test_index_batch_HAPT = HAPT.data_batch(df_en_test_HAPT.index.values, core)

data_train_index_batch_KTH = KTH.data_batch(df_en_train_KTH.index.values, core)
data_pre_train_index_batch_KTH = KTH.data_batch(df_en_pre_train_KTH.index.values, core)
data_validation_index_batch_KTH = KTH.data_batch(df_en_validation_KTH.index.values, core)
data_test_index_batch_KTH = KTH.data_batch(df_en_test_KTH.index.values, core)

data_train_index_batch_NMNIST = NMNIST.data_batch(df_en_train_NMNIST.index.values, core)
data_pre_train_index_batch_NMNIST = NMNIST.data_batch(df_en_pre_train_NMNIST.index.values, core)
data_validation_index_batch_NMNIST = NMNIST.data_batch(df_en_validation_NMNIST.index.values, core)
data_test_index_batch_NMNIST = NMNIST.data_batch(df_en_test_NMNIST.index.values, core)


# --- define network run function ---
def init_net(gen):
    exec(exec_env)
    exec(exec_var)

    # ---- set numpy random state for each run----
    np.random.set_state(np_state)

    # ---- register the gen to the decoder ----
    generator.decoder.register(gen)

    # --- create network ---
    net = Network()
    LSM_network = generator.generate_network()
    generator.initialize(LSM_network)
    generator.join(net, LSM_network)
    net.store('init')
    return net


def run_net(gen, state_pre_run, Training_task, data_indexes):
    exec(exec_env)
    exec(exec_var)

    # ---- choose generator based on training purpose ----
    if Training_task == 'HAPT':
        HAPT_ = HAPT_classification(coding_duration_HAPT)
        df_en_train = HAPT_.load_data(data_path_HAPT + 'Spike_train_Data/train_' + DataName_HAPT + '.p')
        df_en_validation = HAPT_.load_data(data_path_HAPT + 'Spike_train_Data/validation_' + DataName_HAPT + '.p')
        df_en_test = HAPT_.load_data(data_path_HAPT + 'Spike_train_Data/test_' + DataName_HAPT + '.p')
    elif Training_task == 'KTH':
        KTH_ = KTH_classification()
        df_en_train = KTH_.load_data(data_path_KTH + 'Spike_train_Data/train_' + DataName_KTH + '.p')
        df_en_validation = KTH_.load_data(data_path_KTH + 'Spike_train_Data/validation_' + DataName_KTH + '.p')
        df_en_test = KTH_.load_data(data_path_KTH + 'Spike_train_Data/test_' + DataName_KTH + '.p')
    elif Training_task == 'NMNIST':
        NMNIST_ = NMNIST_classification(shape, neurons_encoding_NMNIST)
        df_en_train = NMNIST_.load_data(data_path_NMNIST + 'Spike_train_Data/train_' + DataName_NMNIST + '.p')
        df_en_validation = NMNIST_.load_data(data_path_NMNIST + 'Spike_train_Data/validation_' + DataName_NMNIST + '.p')
        df_en_test = NMNIST_.load_data(data_path_NMNIST + 'Spike_train_Data/test_' + DataName_NMNIST + '.p')

    # --- run network ---
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

    for ind in data_indexes[0]:
        data = df_en_train.value[ind]
        stimulus = TimedArray(data, dt=Dt)
        duration = data.shape[0]
        net.run(duration * Dt)
        state = net.get_states()['block_readout']['v']
        states_train.append(state)
        labels_train.append(df_en_train.label[ind])
        net.restore('pre_run')

    for ind in data_indexes[1]:
        data = df_en_validation.value[ind]
        stimulus = TimedArray(data, dt=Dt)
        duration = data.shape[0]
        net.run(duration * Dt)
        state = net.get_states()['block_readout']['v']
        states_val.append(state)
        labels_val.append(df_en_validation.label[ind])
        net.restore('pre_run')

    for ind in data_indexes[2]:
        data = df_en_test.value[ind]
        stimulus = TimedArray(data, dt=Dt)
        duration = data.shape[0]
        net.run(duration * Dt)
        state = net.get_states()['block_readout']['v']
        states_test.append(state)
        labels_test.append(df_en_test.label[ind])
        net.restore('pre_run')

    return states_train, labels_train, states_val, labels_val, states_test, labels_test


def parallel_run(fun, data):
    data_list = [x for x in data]
    while True:
        try:
            # ------apply the pool-------
            pool = Pool(processes=core, ray_address=ray_cluster_address, maxtasksperchild=None)
            result = pool.map(fun, data_list)
            # ------close the pool-------
            pool.close()
            pool.join()
            return result
        except (RayActorError, WorkerCrashedError) as e:
            print('restart task: ', e)
            cluster.reconnect(cluster.check_alive())
        except RayError as e:
            print('restart task: ', e)
            ray.shutdown()
            cluster.restart()
        except Exception as e:
            print('restart task: ', e)
            ray.shutdown()
            cluster.restart()

@ProgressBar
@Timelog
def parameters_search(**parameter):

    # ---- choose parameter searcher based on training purpose ----
    score_validation_, score_test_, score_train_ = {}, {}, {}
    score_validation_['HAPT'], score_test_['HAPT'], score_train_['HAPT'] = \
        parameters_search_one_task(data_pre_train_index_batch_HAPT, data_train_index_batch_HAPT,
                                   data_test_index_batch_HAPT, Training_task = 'HAPT', **parameter)
    score_validation_['KTH'], score_test_['KTH'], score_train_['KTH'] = \
        parameters_search_one_task(data_pre_train_index_batch_KTH, data_train_index_batch_KTH,
                                   data_test_index_batch_KTH, Training_task = 'KTH', **parameter)
    score_validation_['NMNIST'], score_test_['NMNIST'], score_train_['NMNIST'] = \
        parameters_search_one_task(data_pre_train_index_batch_NMNIST, data_train_index_batch_NMNIST,
                                   data_test_index_batch_NMNIST, Training_task = 'NMNIST', **parameter)
    return mean(score_validation_.values()), mean(score_test_.values()), mean(score_train_.values())


@Timelog
def parameters_search_one_task(data_train_index_batch, data_validation_index_batch, data_test_index_batch,
                               Training_task, **parameter):
    # ------convert the parameter to gen -------
    gen = [parameter[key] for key in decoder.get_keys]
    # ------init net and run for pre_train-------
    state_pre_run = load()
    # ------parallel run for training data-------
    results_list = parallel_run(partial(run_net, gen, state_pre_run, Training_task), zip(data_train_index_batch,
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
    print('task: %s', Training_task)
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

    if method == 'GA':
        coe = CoE(parameters_search, None, decoder.get_SubCom, decoder.get_ranges, decoder.get_borders,
                  decoder.get_precisions, decoder.get_codes, decoder.get_scales, decoder.get_keys,
                  random_state=seed, maxormin=1)
        coe.optimize(recopt=0.9, pm=0.2, MAXGEN=9 + 2, NIND=10, SUBPOP=1, GGAP=0.5,
                     selectStyle='tour', recombinStyle='reclin',
                     distribute=False, load_continue=load_continue)

    elif method == 'GA_rf':
        coe = CoE_surrogate(parameters_search, None, decoder.get_SubCom, decoder.get_ranges, decoder.get_borders,
                            decoder.get_precisions, decoder.get_codes, decoder.get_scales, decoder.get_keys,
                            random_state=seed, maxormin=1,
                            surrogate_type='rf', init_points=100, LHS_path=LHS_path,
                            acq='lcb', kappa=2.576, xi=0.0, n_estimators=100, min_variance=0.0)
        coe.optimize(recopt=0.9, pm=0.2, MAXGEN=450 + 50, NIND=20, SUBPOP=1, GGAP=0.5,
                     online=True, eva=2, interval=10,
                     selectStyle='tour', recombinStyle='reclin',
                     distribute=False, load_continue=load_continue)