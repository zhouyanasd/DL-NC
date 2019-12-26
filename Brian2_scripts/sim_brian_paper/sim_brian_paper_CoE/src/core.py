# -*- coding: utf-8 -*-
"""
    The basic function defined for the simulation.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

import os
import pickle
import time

import numpy as np
import scipy as sp
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from brian2 import NeuronGroup, Synapses


class Timelog():
    def __init__(self, func):
        self.func = func
        self.itime = time.time()
        self.iteration = 0
        with open('Results_Record' + '.dat', 'w') as f:
            f.write('iteration' + ' '
                    + 'wall_time' + ' '
                    + 'result_validation' + ' '
                    + 'result_test' + ' '
                    + 'result_train' + ' '
                    + 'parameters' + ' '
                    + '\n')

    def __call__(self, *args, **kwargs):
        validation, test, train, parameters = self.func(*args, **kwargs)
        self.save(validation, test, train, parameters)
        return validation

    @property
    def elapsed(self):
        return time.time() - self.itime

    def save(self, validation, test, train, parameters):
        self.iteration += 1
        with open('Results_Record' + '.dat', 'a') as f:
            f.write(str(self.iteration) + ' ' + str(self.elapsed) + ' ' + str(validation) + ' '
                    + str(test) + ' ' + str(train) + ' ' + str(parameters) + ' ' + '\n')


class AddParaName():
    def __init__(self, func):
        self.func = func
        self.keys = []

    def __call__(self, *arg, **kwargs):
        if kwargs:
            return self.func(**kwargs)
        if arg:
            kwargs = dict(zip(self.keys, *arg))
            return self.func(**kwargs)


class MathFunctions():
    """
    Some math functions for the simulations

    """
    def __init__(self):
        pass

    def gamma(self, a, size):
        return sp.stats.gamma.rvs(a, size=size)


class BaseFunctions():
    """
    Some basic functions for the simulations

    """

    def __init__(self):
        pass

    def initialize_parameters(object, parameter_name, parameter_value):
        '''
         Set the initial parameters of the objects in the block.

         Parameters
         ----------
         object: Brian2.NeuronGroup or Brian2.Synapse, one of the two kinds of objects.
         parameter_name: str, the name of the parameter.
         parameter_value: np.array, the value of the parameter.
         '''
        if isinstance(object, NeuronGroup):
            object.variables[parameter_name].set_value(parameter_value)
        elif isinstance(object, Synapses):
            object.pre.variables[parameter_name].set_value(parameter_value)
        else:
            print('wrong object type')

    def full_connect_encoding(self, neurons_encoding, reservoir, strength_synapse_encoding_reservoir):
        connect_matrix_encoding = []
        converted_strength_synapse_encoding_reservoir = []
        for block_input_index in reservoir.input:
            block_input = reservoir.blocks[block_input_index]
            connect_matrix = np.meshgrid(np.arange(len(neurons_encoding)), block_input.input)
            connect_matrix_encoding.append(connect_matrix)

            strength_synapse = np.zeros(neurons_encoding, block_input.N)
            strength = strength_synapse_encoding_reservoir[block_input_index]
            strength_ = np.random.rand(len(neurons_encoding) * len(block_input.input)) * strength
            for index, (index_i, index_j) in enumerate(zip(connect_matrix)):
                strength_synapse[index_i][index_j] = strength_[index]
            converted_strength_synapse_encoding_reservoir.append(strength_synapse)
            return connect_matrix_encoding, converted_strength_synapse_encoding_reservoir

    def convert_connect_matrix_reservoir(self, reservoir, strength_synapse_reservoir):
        converted_connect_matrix_reservoir = []
        converted_strength_synapse_reservoir = []
        for index, synapse in enumerate(reservoir.synapses):
            block_pre_index = reservoir.connect_matrix[0][index]
            block_post_index = reservoir.connect_matrix[1][index]
            block_pre = reservoir.blocks[block_pre_index]
            block_post = reservoir.blocks[block_post_index]
            connect_matrix = np.meshgrid(block_pre.output, block_post.input)
            converted_connect_matrix_reservoir.append(connect_matrix)

            strength_synapse = np.zeros(block_pre.N, block_post.N)
            strength = strength_synapse_reservoir[block_pre_index][block_post_index]
            strength_ = np.random.rand(len(block_pre.output) * len(block_post.input))*strength
            for index, (index_i, index_j) in enumerate(zip(connect_matrix)):
                strength_synapse[index_i][index_j] = strength_[index]
            converted_strength_synapse_reservoir.append(strength_synapse)

            return converted_connect_matrix_reservoir, converted_strength_synapse_reservoir


    def get_weight_connection_matrix(self, connection_matrix, weight):
        weight_list = []
        for index_i, index_j in connection_matrix[0], connection_matrix[1]:
            weight_list.append(weight[index_i][index_j])
        return weight_list

    def np_extend(self, a, b, axis=0):
        if a is None:
            shape = list(b.shape)
            shape[axis] = 0
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b, axis)

    def np_append(self, a, b):
        shape = list(b.shape)
        shape.insert(0, -1)
        if a is None:
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b.reshape(tuple(shape)), axis=0)


class Evaluation():
    """
    Some basic function for evaluate the output of LSM.

    This class offers evaluation functions for learning tasks.

    """

    def __init__(self):
        pass

    def readout_sk(self, X_train, X_validation, X_test, y_train, y_validation, y_test, **kwargs):
        lr = LogisticRegression(**kwargs)
        lr.fit(X_train.T, y_train.T)
        y_train_predictions = lr.predict(X_train.T)
        y_validation_predictions = lr.predict(X_validation.T)
        y_test_predictions = lr.predict(X_test.T)
        return accuracy_score(y_train_predictions, y_train.T), \
               accuracy_score(y_validation_predictions, y_validation.T), \
               accuracy_score(y_test_predictions, y_test.T)


class Result():
    """
    Some functions for dealing with the results.

    """
    def __init__(self):
        pass

    def result_save(self, path, *arg, **kwarg):
        if os.path.exists(path):
            os.remove(path)
        fw = open(path, 'wb')
        pickle.dump(kwarg, fw)
        fw.close()

    def result_pick(self, path):
        fr = open(path, 'rb')
        data = pickle.load(fr)
        fr.close()
        return data