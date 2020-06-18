import numpy as np
import scipy as sp
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from brian2 import NeuronGroup, Synapses


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

    def initialize_parameters(self, object, parameter_name, parameter_value):
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

    def get_parameters(self, connection_matrix, parameter):
        parameter_list = []
        for index_i, index_j in connection_matrix[0], connection_matrix[1]:
            parameter_list.append(parameter[index_i][index_j])
        return parameter_list

    def np_two_combination(self, a, b):
        x = []
        y = []
        for i in a:
            for j in b:
                x.append(i)
                y.append(j)
        return np.array[x, y]

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