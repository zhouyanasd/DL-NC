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
        return self.__operation

    def set_operation_on(self):
        self.__operation = True

    def set_operation_off(self):
        self.__operation = False


    def get_global_time(self):
        return self.__global_time

    def set_global_time(self,time):
        self.__global_time = time

    def add_global_time(self,dt = 1):
        self.__global_time = self.__global_time + dt


    def get_global_connection(self):
        return self.__global_connection

    def set_global_connection(self,connection):
        self.__global_connection = connection

    def add_global_connection(self,pre_conn,post_conn,self_conn):
        self.__global_connection = np.concatenate((self.__global_connection,post_conn),axis=0)
        temp_conn = np.concatenate((pre_conn,self_conn), axis=1)
        self.__global_connection = np.concatenate((self.__global_connection,temp_conn.T),axis=1)