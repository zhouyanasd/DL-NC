import numpy as np

from ..core import Base

class Synapse(Base):

    def __init__(self, id, pre_neuron, post_neuron, delay, weight = np.random.normal(0, 1)):
        self.id = id
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = delay

    def register(self):
        self.pre_neuron.post_synapse = np.concatenate((self.pre_neuron.post_synapse,[self]),axis=0)
        self.post_neuron.pre_synapse = np.concatenate((self.post_neuron.pre_synapse,[self]),axis=0)

    def adjust_weight(self):
        pass

    def trans_fired(self):
        pass
