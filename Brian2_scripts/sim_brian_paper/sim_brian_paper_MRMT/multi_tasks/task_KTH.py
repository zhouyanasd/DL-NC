# -*- coding: utf-8 -*-
"""
    The class of network runner for KTH task.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src.config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks.task_basic import *

from brian2 import *

class task_KTH_evaluator(task_evaluator):
    """
    Some basic functions for the network runner.
    The class is build for separating name space of different class.

    """

    def __init__(self):
        super().__init__()
        # -----data path setting-------
        self.data_path = project_dir_sever+'/Data/KTH/'
        self.DataName = 'coe_[15,5,4]'
        self.LHS_path = exec_dir +'/LHS_KTH.dat'

    def init_task(self):

        # -----simulation parameter setting-------
        origin_size = (120, 160)
        pool_size = (5, 5)
        pool_types = 'max'
        pool_threshold = 0.2

        F_train = 1
        F_pre_train = 0.2
        F_validation = 1
        F_test = 1

        self.neurons_encoding = int((origin_size[0] * origin_size[1]) / (pool_size[0] * pool_size[1]))

        ##-----------------------------------------------------
        # --- data and classifier ----------------------
        KTH = KTH_classification()

        # -------data initialization----------------------
        try:
            df_en_train = KTH.load_data(self.data_path + 'train_' + self.DataName+'.p')
            df_en_pre_train = KTH.load_data(self.data_path + 'pre_train_' + self.DataName + '.p')
            df_en_validation = KTH.load_data(self.data_path + 'validation_' + self.DataName+'.p')
            df_en_test = KTH.load_data(self.data_path + 'test_' + self.DataName+'.p')
        except FileNotFoundError:
            KTH.load_data_KTH_all(self.data_path, split_type='mixed', split=[15, 5, 4])

            df_train = KTH.select_data_KTH(F_train, KTH.train, False)
            df_pre_train = KTH.select_data_KTH(F_pre_train, KTH.train, False)
            df_validation = KTH.select_data_KTH(F_validation, KTH.validation, False)
            df_test = KTH.select_data_KTH(F_test, KTH.test, False)

            df_en_train = KTH.encoding_latency_KTH(df_train, origin_size, pool_size, pool_types, pool_threshold)
            df_en_pre_train = KTH.encoding_latency_KTH(df_pre_train, origin_size, pool_size, pool_types, pool_threshold)
            df_en_validation = KTH.encoding_latency_KTH(df_validation, origin_size, pool_size, pool_types, pool_threshold)
            df_en_test = KTH.encoding_latency_KTH(df_test, origin_size, pool_size, pool_types, pool_threshold)

            KTH.dump_data(self.data_path + 'train_' + self.DataName + '.p', df_en_train)
            KTH.dump_data(self.data_path + 'pre_train_' + self.DataName + '.p', df_en_pre_train)
            KTH.dump_data(self.data_path + 'validation_' + self.DataName + '.p', df_en_validation)
            KTH.dump_data(self.data_path + 'test_' + self.DataName + '.p', df_en_test)

        self.data_train_index_batch = KTH.data_batch(df_en_train.index.values, core)
        self.data_pre_train_index_batch = KTH.data_batch(df_en_pre_train.index.values, core)
        self.data_validation_index_batch = KTH.data_batch(df_en_validation.index.values, core)
        self.data_test_index_batch = KTH.data_batch(df_en_test.index.values, core)

    def register_decoder_generator(self, decoder, generator):
        self.decoder = decoder
        self.generator = generator

    # --- define network run function ---
    def pre_run_net(self, gen, data_index):
        exec(exec_env)
        exec(exec_var)

        # --- choose data based on training task ---
        KTH_ = KTH_classification()
        df_en_pre_train = KTH_.load_data(self.data_path + 'pre_train_' + self.DataName + '.p')

        # --- run network ---
        net = self.init_net(gen)
        Switch = 1
        for ind in data_index:
            data = df_en_pre_train.value[ind]
            stimulus = TimedArray(data, dt=Dt)
            duration = data.shape[0]
            net.run(duration * Dt)
        return net._full_state()

    def run_net(self, gen, state_pre_run, data_indexes):
        exec(exec_env)
        exec(exec_var)

        # ---- choose generator based on training purpose ----
        KTH_ = KTH_classification()
        df_en_train = KTH_.load_data(self.data_path + 'Spike_train_Data/train_' + self.DataName + '.p')
        df_en_validation = KTH_.load_data(self.data_path + 'Spike_train_Data/validation_' + self.DataName + '.p')
        df_en_test = KTH_.load_data(self.data_path + 'Spike_train_Data/test_' + self.DataName + '.p')

        # --- run network ---
        net = self.init_net(gen)
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