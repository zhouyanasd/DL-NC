# -*- coding: utf-8 -*-
"""
    The Multi-tasks training of Liquid State Machine
    (LSM). The optimization method adopted here is Genetic algorithm (GA).

    This is used for training single block with specific task.

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
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src.config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks.ray_config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks.sim_config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks import tasks
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks.task_ray import parallel_run

from brian2 import *
from sklearn.preprocessing import MinMaxScaler

import ray
import gc
from functools import partial

ray_cluster_address = 'auto'

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
# -----simulation settings-------
method = 'GA'
total_eva = 600
load_continue = False
task_id = 0

# -----task runner selector-------
task_evaluator = tasks[task_id]['evaluator']()
task_evaluator.init_task()

#--- create generator and decoder ---
decoder = Decoder_Block(config_group_block, config_keys_block, config_SubCom_block, config_codes_block,
                        config_ranges_block, config_borders_block, config_precisions_block, config_scales_block,
                        gen_group_block, task_evaluator.neurons_encoding)
generator = Generator_Block(np_state, task_id = task_id)
generator.register_decoder(decoder)

#--- register generator and decoder ---
task_evaluator.register_generator(generator)

# -----classifier-------
accuracy_evaluator = Evaluation()

# -----parameters search for optimizer-------
@ProgressBar
@Timelog
def parameters_search(**parameter):
    # ------convert the parameter to gen -------
    gen = [parameter[key] for key in task_evaluator.generator.decoder.get_keys]
    # ------init net and run for pre_train-------
    net_state_list = parallel_run(cluster, partial(task_evaluator.pre_run_net, gen),
                                                 task_evaluator.data_pre_train_index_batch)
    state_pre_run = task_evaluator.sum_strength(gen, net_state_list)
    # ------parallel run for training data-------
    results_list = parallel_run(cluster, partial(task_evaluator.run_net, gen, state_pre_run),
                                               zip(task_evaluator.data_train_index_batch,
                                                   task_evaluator.data_validation_index_batch,
                                                   task_evaluator.data_test_index_batch))
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
    score_train, score_validation, score_test = \
        accuracy_evaluator.readout_sk(states_train, states_validation, states_test,
                                      np.asarray(_label_train),np. asarray(_label_validation), np.asarray(_label_test),
                                      solver="lbfgs", multi_class="multinomial")
    # ------collect the memory-------
    gc.collect()
    # ------save best state---------
    if is_save_state and task_evaluator.is_best_test(score_test):
        task_evaluator.dump_data(Optimal_state + tasks[task_id]['name'] + '.pkl', state_pre_run)
        task_evaluator.dump_data(Optimal_gens + tasks[task_id]['name'] + '.pkl', gen)
    # ----------show results-----------
    print('task: ', tasks[task_evaluator.generator.task_id]['name'])
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
    parameters_search.file_name = tasks[task_id]['name']
    parameters_search.func.file_name = tasks[task_id]['name']

    # -------parameters search---------------
    if method == 'GA':
        optimizer = CoE(parameters_search, None, decoder.get_SubCom, decoder.get_ranges, decoder.get_borders,
                        decoder.get_precisions, decoder.get_codes, decoder.get_scales, decoder.get_keys,
                        random_state=seeds, maxormin=1)
        optimizer.optimize(recopt=0.9, pm=0.2, MAXGEN=9 + 2, NIND=10, SUBPOP=1, GGAP=0.5,
                           selectStyle='tour', recombinStyle='reclin',
                           distribute=False, load_continue=load_continue)

    elif method == 'GA_rf':
        optimizer = CoE_surrogate(parameters_search, None, decoder.get_SubCom, decoder.get_ranges, decoder.get_borders,
                                  decoder.get_precisions, decoder.get_codes, decoder.get_scales, decoder.get_keys,
                                  random_state=seeds, maxormin=1,
                                  surrogate_type='rf', init_points=100, LHS_path=task_evaluator.LHS_path,
                                  acq='lcb', kappa=2.576, xi=0.0, n_estimators=100, min_variance=0.0)
        optimizer.optimize(recopt=0.9, pm=0.2, MAXGEN=450 + 50, NIND=20, SUBPOP=1, GGAP=0.5,
                           online=True, eva=2, interval=10,
                           selectStyle='tour', recombinStyle='reclin',
                           distribute=False, load_continue=load_continue)