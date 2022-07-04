# -*- coding: utf-8 -*-
"""
    The fundamental class of network runner for tasks.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks.sim_config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src import BaseFunctions

from brian2 import *

import ray
from ray.util.multiprocessing import Pool
from ray.exceptions import RayActorError, WorkerCrashedError, RayError

class task_evaluator(BaseFunctions):
    """
    Some basic functions for the network runner.
    The class is build for separating name space of different class.

    """

    def __init__(self):
        self.best_test = 1

    def is_best_test(self, test):
        if test < self.best_test:
            self.best_test = test
            return True
        else:
            return False

    def init_net(self, gen):
        exec(exec_env)
        exec(exec_var)

        # ---- set numpy random state for each run----
        np.random.set_state(np_state)

        # ---- register the gen to the decoder ----
        self.generator.decoder.register(gen)

        # --- create network ---
        net = Network()
        LSM_network = self.generator.generate_network()
        self.generator.initialize(LSM_network)
        self.generator.join(net, LSM_network)
        net.store('init')
        return net

    def sum_strength(self, gen, net_state_list):
        net = self.init_net(gen)
        state_init = net._full_state()
        l = len(net_state_list)
        for com in list(state_init.keys()):
            if 'block_block_' in com or 'pathway_' in com and '_pre' not in com and '_post' not in com:
                try:
                    para_init = list(state_init[com]['strength'])
                    np.subtract(para_init[0], para_init[0], out=para_init[0])
                    for state in net_state_list:
                        para = list(state[com]['strength'])
                        np.add(para_init[0], para[0] / l, out=para_init[0])
                    state_init[com]['strength'] = tuple(para_init)
                except:
                    continue
        return state_init

    def parallel_run(self, cluster, fun, data):
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