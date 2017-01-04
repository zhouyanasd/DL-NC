import numpy as np

from ..core import Base,MAX_WEIGHT,MIN_WEIGHT
from ..function import Plasticity

class Synapse(Base):

    def __init__(self, id, pre_neuron, post_neuron, delay, plasticity = 'STDP',weight = np.random.normal(0, 1)):
        self.id = id
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = delay
        self.d_w = 0
        self.plasticity = getattr(Plasticity(),plasticity)
        self.spiking_buffer = np.zeros(self.delay)                                                                      # the index = 0 is out and index=max is in
        self._last_arrive_time = 0
        self._last_spiking_time = 0

    def register(self):
        self.pre_neuron.post_synapse = np.concatenate((self.pre_neuron.post_synapse,[self]),axis=0)
        self.post_neuron.pre_synapse = np.concatenate((self.post_neuron.pre_synapse,[self]),axis=0)


    def adjust_weight(self):
        pass

    def update(self):
        self.weight += self.d_w
        self.d_w = 0

        if (self.pre_neuron.type == 0):
            if (self.weight > MAX_WEIGHT):
                self.weight = MAX_WEIGHT
            if (self.weight < 0.0):
                self.weight = 0.0

        if (self.pre_neuron.type > 0):
            if (self.weight < MIN_WEIGHT):
                self.weight = MIN_WEIGHT
            if (self.weight > 0.0):
                self.weight = 0.0

    def trans_fired(self):
        self.spiking_buffer[0:self.delay-1] = self.spiking_buffer[1:self.delay]
        if self.pre_neuron.fired:
            self.spiking_buffer[self.delay-1] = 1
        else:
            self.spiking_buffer[self.delay-1] = 0

    def reset(self):
        pass

