import numpy as np
from. import global_constant

class Base(object):

    '''__global_connection is the adjacency matrix of a digraph where "0" and "1" represent
    no connection and be connected by other, respectively. For instance, in the matrix ([0,1,0],
    [1,0,1],[1,0,0]), means neuron '0' connects to neuron '1', neuron '1' connect to neuron '0'
    and '2'.
    '''
    __operation = False
    __global_time = 0
    __global_connection = np.zeros((1,1),dtype =global_constant.CONNECTION_ARRAY_DTYPE)


    def get_operation(self):
        return Base.__operation

    def set_operation_on(self):
        Base.__operation = True

    def set_operation_off(self):
        Base.__operation = False


    def get_global_time(self):
        return Base.__global_time

    def set_global_time(self,time):
        Base.__global_time = time

    def add_global_time(self,dt = 1):
        Base.__global_time = Base.__global_time + dt


    def get_global_connection(self):
        return Base.__global_connection

    def set_global_connection(self,connection):
        Base.__global_connection = connection

    def add_global_connection(self,pre_conn,post_conn,self_conn):
        Base.__global_connection = np.concatenate((Base.__global_connection,post_conn),axis=0)
        temp_conn = np.concatenate((pre_conn,self_conn), axis=1)
        Base.__global_connection = np.concatenate((Base.__global_connection,temp_conn.T),axis=1)