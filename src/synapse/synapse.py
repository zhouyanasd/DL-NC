from ..core import Base

class Synapse(Base):

    def __init__(self, pre_neuron, post_neuron, weight, delay):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self .delay = delay