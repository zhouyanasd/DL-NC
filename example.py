import numpy as np
import theano.tensor as T
import theano

# x = T.dscalar('x')
# y = T.dscalar('y')
# z = x + y
# f = function([x,y],z)
#
# print(f(2,3))
#
# from theano import pp
# print(pp(z))
#
# x=T.dmatrix('x')
# y=T.dmatrix('y')
# z = x+y
# f = function([x,y],z)
# print(f(np.arange(12).reshape((3,4)),10*np.ones((3,4))))

class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function = None):
        self.W = theano.shared(np.random.normal(0,1,(in_size,out_size)))
        self.b = theano.shared(np.zeros((out_size,))+0.1)
        self.Wx_plus_b = T.dot(inputs,self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)

