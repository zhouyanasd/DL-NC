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
    def __init__(self):
        pass

    def gamma(self, a, size):
        return sp.stats.gamma.rvs(a, size=size)


class BaseFunctions():
    """Standardize a dataset along any axis

    Center to the mean and component wise scale to unit variance.

    Notes
    -----

    See also
    --------
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

    def allocate(self, G, X, Y, Z):
        V = np.zeros((X, Y, Z), [('x', float), ('y', float), ('z', float)])
        V['x'], V['y'], V['z'] = np.meshgrid(np.linspace(0, Y - 1, Y), np.linspace(0, X - 1, X),
                                             np.linspace(0, Z - 1, Z))
        V = V.reshape(X * Y * Z)
        np.random.shuffle(V)
        n = 0
        for g in G:
            for i in range(g.N):
                g.x[i], g.y[i], g.z[i] = V[n][0], V[n][1], V[n][2]
                n += 1
        return G


class Readout():
    """Some basic function for data transformation or calculation.

    This class offers ....

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    See also
    --------

    Notes
    -----

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
