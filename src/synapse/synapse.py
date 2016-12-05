from ..core import Base

class Synapse(Base):

    def __init__(self, id, pre_neuron, post_neuron, weight, delay):
        self.id = id
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = delay

    def register(self):
        self.pre_neuron.post_synapse =

    def adjust_weight(self):
        pass
