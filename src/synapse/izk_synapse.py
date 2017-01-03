from .import Synapse

import numpy as np

class izk_synapse(Synapse):
    def __init__(self, id, pre_neuron, post_neuron, delay, weight):
        Synapse.__init__( id, pre_neuron, post_neuron, delay, weight)

    def adjust_weight(self):
        pass



