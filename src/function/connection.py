from src.core import Base

import numpy as np

class Connnection(Base):

    def __init__(self):
        pass

    def conn_normal(self,neuron_list):
        conn = np.arange(neuron_list.size)
        np.random.shuffle(conn)
        conn = conn[:2]
        pre_neuron = neuron_list[conn[0]]
        post_neuron = neuron_list[conn[1]]
        return pre_neuron,post_neuron

    def conn_izk(self,neuron_list):
        conn = np.arange(neuron_list.size)
        np.random.shuffle(conn)
        pre_neuron = neuron_list[conn[0]]
        post_neuron = neuron_list[conn[1]]
        if pre_neuron.type == 0:
            i = 1
            while (post_neuron.type == 0):
                post_neuron = neuron_list[conn[1+i]]
        return pre_neuron,post_neuron

