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

    Functions
    ----------

    """
    def __init__(self, N, equ, on_pre):
        self.N = N

    def create_neurons(self, model, threshold, reset,refractory):
        self.neurons = NeuronGroup(self.N, self.equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms )

    def create_synapse(self, ):
        self.synapse = Synapses(self.neurons, self.neurons, 'w = 1 : 1',on_pre = self.on_pre, method='linear', delay = 1*ms)

    def connect(self, connect_matrix):
        self.synapse.connect(i=[],j=[])

    def join_networks(self, net):
        net.add(self.neurons, self.synapse)


class Neuron():
    """Some basic function for data transformation or calculation.

    This class offers ....

    Parameters
    ----------
    property: 'ex' or 'inh'

    """
    def __init__(self, property):
        self.property = property