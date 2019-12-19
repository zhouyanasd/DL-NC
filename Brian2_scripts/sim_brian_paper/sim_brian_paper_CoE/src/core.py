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