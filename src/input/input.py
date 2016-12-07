import numpy as np

from ..core import Base
from ..neuron import SpikingNeuron as Neuron

class Input(Base):
    def __init__(self, input_size, conn_neuron_n):
        self.in_size = input_size
        self.conn_neuron_n = conn_neuron_n
        self.conn_neuron = np.zeros((self.in_size,self.conn_neuron_n), dtype =np.dtype([('neuron', Neuron),('weight',np.float64)]))

    def initialization(self):
        pass

    def register(self):
        pass

    def Possion(self):
        pass

