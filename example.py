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
    def __init__(self):
