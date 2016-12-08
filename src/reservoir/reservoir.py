import numpy as np

from ..core import Base
from ..neuron import SpikingNeuron as Neuron
from ..synapse import Synapse

class Reservoir(Base):
    def __init__(self,n_type,r_size,s_type,s_number):
        self.r_size = r_size
        self.s_number = s_number
        self.neuron_list = np.array([], dtype =np.dtype([('neuron', Neuron)]))
        self.synapse_list = np.array([], dtype= np.dtype ([('synapse',Synapse)]))

    def initialization(self):
        pass

    def connect(self):
        pass







