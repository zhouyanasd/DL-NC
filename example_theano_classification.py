import numpy as np
import theano
import theano.tensor as T

def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict,y_target)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy