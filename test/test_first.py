# import numpy as np
# from src.core import global_constant
# from src.core import Base
#
# base = Base()
# print(base.get_global_connection())
# pre_conn = np.array([[(1,0.5)]],dtype = global_constant.CONNECTION_ARRAY_DTYPE)
# post_conn = np.array([[(0,0.0)]],dtype = global_constant.CONNECTION_ARRAY_DTYPE)
# self_conn = np.array([[(0,0.0)]],dtype = global_constant.CONNECTION_ARRAY_DTYPE)
#
# base.add_global_connection(pre_conn,post_conn,self_conn)
#
# print(base.get_global_connection())

class GenericRepo(object):
    __repo__ = {}

    def fun(self):
        pass

    def get(self):
        for key in GenericRepo.__repo__.keys():
            print (key, GenericRepo.__repo__[key])


class A(GenericRepo):
    def __new__(cls):
        if not hasattr(A, "__inst__"):
            cls.__inst__ = super(GenericRepo, cls).__new__(cls)

        return cls.__inst__

    def fun(self):
        GenericRepo.__repo__["A"] = 1


class B(GenericRepo):
    def __new__(cls):
        if not hasattr(B, "__inst__"):
            cls.__inst__ = super(GenericRepo, cls).__new__(cls)

        return cls.__inst__

    def fun(self):
        GenericRepo.__repo__["B"] = 2

if __name__ == '__main__':
    a = A()
    b = B()

    a.fun()
    b.fun()
    a.get()