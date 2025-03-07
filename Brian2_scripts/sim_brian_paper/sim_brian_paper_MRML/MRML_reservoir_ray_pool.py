# -*- coding: utf-8 -*-
"""
    The Multi-tasks training of Liquid State Machine
    (LSM). The optimization method adopted here is Genetic algorithm (GA).

    This is used for training reservoir with all tasks.

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
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRML.src import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRML.src.config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRML.multi_tasks.ray_config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRML.multi_tasks.sim_config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRML.multi_tasks import tasks
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRML.multi_tasks.task_ray import parallel_run

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
method = 'GA_rf'
total_eva = 600
load_continue = False
LHS_path_reservoir = exec_dir + '/LHS_reservoir.dat'

evaluates = 0
NIND = {'Pre': 2, 'GA': 10, 'GA_rf':20}
N_LHS = 50

last_test = dict(zip(tasks.keys(),[0.0]*len(tasks.keys())))

# -----task runner selector-------
task_evaluators = {}
for task_id, task in tasks.items():
    task_evaluator = tasks[task_id]['evaluator']()
    task_evaluator.init_task()
    task_evaluators[task_id] = task_evaluator

#--- create generator and decoder ---
decoder = Decoder_Reservoir(config_group_reservoir, config_keys_reservoir, config_SubCom_reservoir,
                            config_codes_reservoir, config_ranges_reservoir, config_borders_reservoir,
                            config_precisions_reservoir, config_scales_reservoir, gen_group_reservoir)
generator = Generator_Reservoir(np_state, block_init=block_init, block_max=block_max)
generator.register_decoder(decoder)

#--- initial reservoir generator and decoder ---
optimal_block_gens, state_pre_runs, neurons_encoding = {}, {}, {}
for task_id in tasks.keys():
    optimal_block_gen = decoder.load_data(Optimal_gens + tasks[task_id]['name'] + '.pkl')
    state_pre_run = decoder.load_data(Optimal_state + tasks[task_id]['name'] + '.pkl')
    optimal_block_gens[task_id] = optimal_block_gen
    state_pre_runs[task_id] = state_pre_run
    neurons_encoding[task_id] = task_evaluators[task_id].neurons_encoding
decoder.register_optimal_block_gens(optimal_block_gens)
generator.register_block_generator(neurons_encoding=neurons_encoding)
generator.initialize_task_ids()

#--- register generator and decoder ---
for task_evaluator in task_evaluators.values():
    task_evaluator.register_generator(generator)

# -----classifier-------
accuracy_evaluator = Evaluation()

# -----parameters search for optimizer-------
@ProgressBar
@Timelog
def parameters_search_multi_task(**parameter):
    global evaluates, last_test
    score_validation_, score_test_, score_train_ = {}, {}, {}
    parameters_search.iteration += 1
    for task_id, task_evaluator in task_evaluators.items():
        parameters_search.file_name = tasks[task_id]['name']
        generator.mark_current_task(task_id)
        score_validation_[task_id], score_test_[task_id], score_train_[task_id], parameter_ = \
            parameters_search(task_evaluator, **parameter)
    evaluates += 1
    if len(generator.tasks_ids) < block_max and evaluates >= NIND['Pre']:
        diff = dict(zip(tasks.keys(),
                                   [x-y if x-y>0 else 0.0 for x, y in
                                    zip(score_test_.values(),last_test.values())]))
        p_evaluation = dict(zip(diff.keys(),
                        [(max(diff.values())-x)/sum([max(diff.values())-x for x in diff.values()])
                         for x in diff.values()]))
        last_test = score_test_
        p = np.array(list(p_evaluation.values()))
        task_add = np.random.choice(list(tasks.keys()), p = p.ravel())
        generator.increase_block_reservoir(task_add)
        optimizer.ranges = decoder.get_ranges
        optimizer.codes = decoder.get_codes
        evaluates = 0
    score_validation, score_test, score_train = mean(list(score_validation_.values())), \
                                                mean(list(score_test_.values())), \
                                                mean(list(score_train_.values()))
    # ----------show results-----------
    print('task average: ')
    print('Train score: ', score_train)
    print('Validation score: ', score_validation)
    print('Test score: ', score_test)
    return score_validation, score_test, score_train, parameter

@Timelog
def parameters_search(task_evaluator, **parameter):
    # ------convert the parameter to gen -------
    gen = [parameter[key] for key in task_evaluator.generator.decoder.get_keys]
    # ------init net and run for pre_train-------
    state_pre_run = task_evaluator.get_state_pre_run(gen, state_pre_runs)
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
    # ----------show results-----------
    print('task: ', tasks[task_evaluator.task_id]['name'])
    print('parameter %s' % parameter)
    print('Train score: ', 1-score_train)
    print('Validation score: ', 1-score_validation)
    print('Test score: ', 1-score_test)
    return 1 - score_validation, 1 - score_test, 1 - score_train, parameter

##########################################
# -------optimizer settings---------------
if __name__ == '__main__':
    parameters_search.return_all = True
    parameters_search.auto_increase = False
    parameters_search_multi_task.total = total_eva
    parameters_search_multi_task.load_continue = load_continue
    parameters_search_multi_task.func.load_continue = load_continue
    parameters_search_multi_task.file_name = 'reservoir'
    parameters_search_multi_task.func.file_name = 'reservoir'

    # -------parameters search---------------
    if method == 'GA':
        print('------------ increasing blocks --------------------------')
        optimizer = CoE(parameters_search_multi_task, None, decoder.get_SubCom, decoder.get_ranges, decoder.get_borders,
                        decoder.get_precisions, decoder.get_codes, decoder.get_scales, decoder.get_keys,
                        random_state=seeds, maxormin=1)
        optimizer.optimize(recopt=0.9, pm=0.6, MAXGEN=block_max-block_init-1, NIND=NIND['Pre'], SUBPOP=1, GGAP=0.5,
                       selectStyle='tour', recombinStyle='reclin',
                       distribute=False, load_continue=False)
        print('------------ officially optimizing ----------------------')
        optimizer = CoE(parameters_search_multi_task, None, decoder.get_SubCom, decoder.get_ranges, decoder.get_borders,
                        decoder.get_precisions, decoder.get_codes, decoder.get_scales, decoder.get_keys,
                        random_state=seeds, maxormin=1)
        optimizer.optimize(recopt=0.9, pm=0.6, MAXGEN=29 + 2, NIND=NIND['GA'], SUBPOP=1, GGAP=0.5,
                           selectStyle='tour', recombinStyle='reclin',
                           distribute=False, load_continue=load_continue)

    elif method == 'GA_rf':
        print('------------ increasing blocks ---------------')
        optimizer = CoE(parameters_search_multi_task, None, decoder.get_SubCom, decoder.get_ranges, decoder.get_borders,
                    decoder.get_precisions, decoder.get_codes, decoder.get_scales, decoder.get_keys,
                    random_state=seeds, maxormin=1)
        optimizer.optimize(recopt=0.9, pm=0.6, MAXGEN=block_max-block_init-1, NIND=NIND['Pre'], SUBPOP=1, GGAP=0.5,
                       selectStyle='tour', recombinStyle='reclin',
                       distribute=False, load_continue=False)
        print('------------ officially optimizing ---------------')
        optimizer = CoE_surrogate(parameters_search_multi_task, None, decoder.get_SubCom, decoder.get_ranges, decoder.get_borders,
                                  decoder.get_precisions, decoder.get_codes, decoder.get_scales, decoder.get_keys,
                                  random_state=seeds, maxormin=1,
                                  surrogate_type='rf', init_points=N_LHS, LHS_path=LHS_path_reservoir,
                                  acq='lcb', kappa=2.576, xi=0.0, n_estimators=100, min_variance=0.0)
        optimizer.optimize(recopt=0.9, pm=0.6, MAXGEN=1450 + 50, NIND=NIND['GA_rf'], SUBPOP=1, GGAP=0.5,
                           online=True, eva=2, interval=10,
                           selectStyle='tour', recombinStyle='reclin',
                           distribute=False, load_continue=load_continue)