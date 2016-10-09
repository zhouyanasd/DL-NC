import numpy as np
import theano
import theano.tensor as T


def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction) / len(correct_prediction)
    return accuracy


rng = np.random

N = 400
feats = 784

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

# Declare Theano symbolic variables
x = T.dmatrix('x')
y = T.dvector('y')

#initialize the weights and biases
W = theano.shared(rng.randn(feats), name= 'w')
b = theano.shared(0.1,name='b')

#Construct Theano expression graph
p_1 = T.nnet.sigmoid(T.dot(x,W)+b)
prediction = p_1>0.5
xent = -y*T.log(p_1)-(1-y)*T.log(1-p_1)  #cost
cost = xent.mean()
gW,gb = T.grad(cost,[W,b])
