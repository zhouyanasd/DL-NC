# -*- coding: utf-8 -*-
"""
    The class of network runner for NMNIST task.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src.config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks.task_basic import *

from brian2 import *
from sklearn.model_selection import train_test_split

class task_NMNIST_evaluator(task_evaluator):
    """
    Some basic functions for the network runner.
    The class is build for separating name space of different class.

    """

    def __init__(self):
        super().__init__()
        # -----data path setting-------
        self.data_path = project_dir_sever + '/Data/N_MNIST/'
        self.DataName = 'coe_0.05'
        self.LHS_path = exec_dir + '/LHS_NMNIST.dat'

    def init_task(self):

        # -----simulation parameter setting-------
        self.coding_duration = 10
        self.shape = (34, 34)

        F_train = 0.05833333
        F_pre_train = 0.002
        F_validation = 0.1666667
        F_test = 0.05

        self.neurons_encoding = 2 * self.shape[0] * self.shape[1]

        ##-----------------------------------------------------
        # --- data and classifier ----------------------
        NMNIST = NMNIST_classification(self.shape, self.coding_duration)

        # -------data initialization----------------------
        try:
            self.df_en_train = NMNIST.load_data(self.data_path + 'train_' + self.DataName+'.p')
            self.df_en_pre_train = NMNIST.load_data(self.data_path + 'pre_train_' + self.DataName + '.p')
            self.df_en_validation = NMNIST.load_data(self.data_path + 'validation_' + self.DataName+'.p')
            self.df_en_test = NMNIST.load_data(self.data_path + 'test_' + self.DataName+'.p')
        except FileNotFoundError:
            NMNIST.load_data_NMNIST_all(self.data_path)

            df_train_validation = NMNIST.select_data_NMNIST(F_train, NMNIST.train, False)
            df_pre_train = NMNIST.select_data_NMNIST(F_pre_train, NMNIST.train, False)
            df_train, df_validation = train_test_split(df_train_validation, test_size=F_validation, random_state=42)
            df_test = NMNIST.select_data_NMNIST(F_test, NMNIST.test, False)

            self.df_en_train = NMNIST.encoding_latency_NMNIST(df_train)
            self.df_en_pre_train = NMNIST.encoding_latency_NMNIST(df_pre_train)
            self.df_en_validation = NMNIST.encoding_latency_NMNIST(df_validation)
            self.df_en_test = NMNIST.encoding_latency_NMNIST(df_test)

            NMNIST.dump_data(self.data_path + 'train_' + self.DataName + '.p', self.self.df_en_train)
            NMNIST.dump_data(self.data_path + 'pre_train_' + self.DataName + '.p', self.df_en_pre_train)
            NMNIST.dump_data(self.data_path + 'validation_' + self.DataName + '.p', self.df_en_validation)
            NMNIST.dump_data(self.data_path + 'test_' + self.DataName + '.p', self.df_en_test)

        self.data_train_index_batch = NMNIST.data_batch(self.df_en_train.index.values, cores)
        self.data_pre_train_index_batch = NMNIST.data_batch(self.df_en_pre_train.index.values, cores)
        self.data_validation_index_batch = NMNIST.data_batch(self.df_en_validation.index.values, cores)
        self.data_test_index_batch = NMNIST.data_batch(self.df_en_test.index.values, cores)

    def register_decoder_generator(self, decoder, generator):
        self.decoder = decoder
        self.generator = generator

    # --- define network run function ---
    def pre_run_net(self, gen, data_index):
        exec(exec_env)
        exec(exec_var)

        # --- choose data based on training task ---
        NMNIST_ = NMNIST_classification(self.shape, self.coding_duration)
        df_en_pre_train = NMNIST_.load_data(self.data_path + '/pre_train_' + self.DataName + '.p')

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
        NMNIST_ = NMNIST_classification(self.shape, self.coding_duration)
        df_en_train = NMNIST_.load_data(self.data_path + '/train_' + self.DataName + '.p')
        df_en_validation = NMNIST_.load_data(self.data_path + '/validation_' + self.DataName + '.p')
        df_en_test = NMNIST_.load_data(self.data_path + '/test_' + self.DataName + '.p')

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