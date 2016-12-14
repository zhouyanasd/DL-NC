import numpy as np

from ..core import Base

class Synapse(Base):

    def __init__(self, id, pre_neuron, post_neuron, delay, weight = np.random.normal(0, 1)):
        self.id = id
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = delay
        self.spiking_buffer = np.zeros(self.delay)                                                                      # the index = 0 is out and index=max is in

    def register(self):
        self.pre_neuron.post_synapse = np.concatenate((self.pre_neuron.post_synapse,[self]),axis=0)
        self.post_neuron.pre_synapse = np.concatenate((self.post_neuron.pre_synapse,[self]),axis=0)

    def adjust_weight(self):
        pass

    def trans_fired(self):
        self.spiking_buffer[0:self.delay-1] = self.spiking_buffer[1:self.delay]
        if self.pre_neuron.fired:
            self.spiking_buffer[self.delay-1] = 1
        else:
            self.spiking_buffer[self.delay-1] = 0


    def reset(self):
        pass

