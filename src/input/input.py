import numpy as np

from ..core import Base,INPUT_CONN_RATE
import src.neuron.neuron as Neuron

class Input(Base):
    def __init__(self, input_size, reservoir):
        self.in_size = input_size
        self.input_t = np.zeros(self.in_size)
        self.reservoir = reservoir
        self.conn_neuron = np.zeros((self.in_size, self.reservoir.r_size), dtype = np.dtype([('neuron', Neuron.SpikingNeuron),('weight',np.float64)]))

    def __select_neuron_random(self):
        conn_n = np.random.randint(0,self.reservoir.r_size*INPUT_CONN_RATE,1)[0]                # get the connection number to reservoir for each input
        if conn_n <= 0:                                                                         # connection number greater than 0
            conn_n = 1
        conn = np.arange(self.reservoir.r_size)                                                 # random selected
        np.random.shuffle(conn)
        return conn[:conn_n]


    def initialization(self):
        for i in range(self.in_size):                                                           # for each input
            conn = self.__select_neuron_random()                                                # select neuron to connect randomly
            for j in conn:
                w = np.random.uniform(0,1)                                                      # uniform random weight
                self.conn_neuron[i,j] = (self.reservoir.neuron_list[j],w)
                self.register(i,j)                                                              # register the input i to the neuron id = j


    def register(self,input_index,neuron_id):
        input_conn = np.array([(self,input_index)],dtype = np.dtype([('input', Input),('index',np.int64)]))
        self.reservoir.neuron_list[neuron_id].input = \
            np.concatenate((self.reservoir.neuron_list[neuron_id].input,input_conn),axis=0)
