# -*- coding: utf-8 -*-
"""
    The fundamental class of network runner for tasks.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks.sim_config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src import BaseFunctions

from brian2 import *
import re

class task_evaluator(BaseFunctions):
    """
    Some basic functions for the network runner.
    The class is build for separating name space of different class.

    """

    def __init__(self):
        self.best_test = 1

    def register_generator(self, generator):
        self.generator = generator

    def get_task_id(self):
        from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks import tasks
        for task_id, task in tasks.items():
            if isinstance(self, task['evaluator']):
                return task_id

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

    def find_task_id(self, name):
        try:
            task_id = int(re.findall(r"task(\d+?)_", name)[0])
            return task_id
        except:
            pass

    def get_state_pre_run(self, gen, state_pre_runs):
        net = self.init_net(gen)
        state_init = net._full_state()
        block_block = {}
        pathway_encoding = {}
        for task_id, state_pre_run in state_pre_runs.items():
            for com in list(state_pre_run.keys()):
                if 'block_block_' in com and '_pre' not in com and '_post' not in com and 'encoding'not in com \
                        and 'readout'not in com:
                    block_block[task_id] = state_pre_run[com]
                if 'pathway_encoding_' in com and '_pre' not in com and '_post' not in com:
                    pathway_encoding[task_id] = state_pre_run[com]
        for com in list(state_init.keys()):
            task_id = self.find_task_id(com)
            if 'block_block_' in com and '_pre' not in com and '_post' not in com and 'encoding' not in com \
                    and 'readout' not in com:
                state_init[com] = block_block[task_id]
            if 'pathway_encoding_' in com and '_pre' not in com and '_post' not in com:
                task_id_ = self.find_task_id(net[com].target.name)
                if task_id == task_id_:
                    state_init[com] = pathway_encoding[task_id]
        return state_init