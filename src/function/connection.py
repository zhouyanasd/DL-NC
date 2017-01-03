from src.core import Base, MAX_SYNAPSE_DELAY

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
        delay = np.random.randint(1,MAX_SYNAPSE_DELAY+1)
        if pre_neuron.type == 0:
            weight = 0 - np.random.normal(0, 1)
        else:
            weight = np.random.normal(0, 1)
        return pre_neuron,post_neuron,delay,weight


    def conn_izk(self,neuron_list):
        conn = np.arange(neuron_list.size)
        np.random.shuffle(conn)
        pre_neuron = neuron_list[conn[0]]
        post_neuron = neuron_list[conn[1]]
        if pre_neuron.type == 0:
            i = 1
            while (post_neuron.type == 0):
                post_neuron = neuron_list[conn[1+i]]
            delay = 1
            weight = -5.0
        else:
            delay = np.random.randint(1,MAX_SYNAPSE_DELAY+1)
            weight = 6.0
        return pre_neuron,post_neuron,delay,weight
