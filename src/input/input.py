import numpy as np

from ..core import Base
from ..neuron import SpikingNeuron as Neuron

class Input(Base):
    def __init__(self,ex_input, conn_neuron_n):
        self.in_size = ex_input.size()
        self.conn_neuron_n = conn_neuron_n
        self.conn_neuron = np.zeros((self.in_size,self.conn_neuron_n), dtype =np.dtype([('neuron', Neuron),('weight',np.float64)]))

    def initialization(self,ex_input):
        self.in_size = ex_input.size()

    def register(self):
        pass

    def Possion(self):
        pass

