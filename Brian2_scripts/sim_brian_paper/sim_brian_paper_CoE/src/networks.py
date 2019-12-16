# -*- coding: utf-8 -*-
"""
    The fundamental neurons and network structure
    including local blocks.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""


from brian2 import *

class Block():
    """Some basic function for data transformation or calculation.

    This class offers a basic property and functions of block in LSM.

    Parameters
    ----------
    N: int, the number of neurons
    """

    def __init__(self, N):
        self.N = N

    def create_neurons(self, model, threshold, reset,refractory, **kwargs):
        '''
         Create neurons group for the block.

         Parameters
         ----------
         The parameters follow the necessary 'NeuronGroup' class of Brain2.
         '''
        self.neurons = NeuronGroup(self.N, model, threshold=threshold, reset=reset, refractory=refractory,
                                   method='euler', **kwargs)

    def create_synapse(self, model, on_pre, delay, **kwargs):
        '''
         Create synapse between neurons for the block.

         Parameters
         ----------
         The parameters follow the necessary 'Synapses' class of Brain2.
         '''
        self.synapse = Synapses(self.neurons, self.neurons, model, on_pre = on_pre, delay = delay,
                                method='euler', **kwargs)

    def connect(self, connect_matrix):
        '''
         Connect neurons using synapse based on the fixed connection matrix.

         Parameters
         ----------
         connect_matrix: numpy array, the fixed connection matrix.
         '''
        self.synapse.connect(i = connect_matrix[0], j = connect_matrix[1])

    def join_networks(self, net):
        '''
         Let the objects of block join the whole neural network.

         Parameters
         ----------
         net: Brian2 Network object, the existing neural network.
         '''
        net.add(self.neurons, self.synapse)


# class Neuron():
#     """Some basic function for data transformation or calculation.
#
#     This class offers ....
#
#     Parameters
#     ----------
#     property: basestring, 'ex' or 'inh'
#     """
#
#     def __init__(self, property):
#         self.property = property