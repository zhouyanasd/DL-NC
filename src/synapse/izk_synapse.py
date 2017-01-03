from .import Synapse

import numpy as np

class izk_synapse(Synapse):
    def __init__(self,id, pre_neuron, post_neuron, delay, weight = np.random.normal(0, 1)):
        Synapse.__init__(id, pre_neuron, post_neuron, delay, weight)



