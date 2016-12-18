import numpy as np
import src

from ..core import Base, MAX_SYNAPSE_DELAY,INTER_RESERVOIR_CONN_RATE

class Reservoir(Base):
    def __init__(self,id,r_size,s_number,n_type = 'SpikingNeuron', s_type ='Synapse'):
        self.id = id
        self.r_size = r_size
        self.neu_class = getattr(src.neuron,n_type)
        self.s_number = (np.ceil(s_number*INTER_RESERVOIR_CONN_RATE)).astype(int)
        self.syn_class = getattr(src.synapse,s_type)
        self.neuron_list = np.array([], dtype =np.dtype([('neuron', self.neu_class)]))
        self.synapse_list = np.array([], dtype= np.dtype ([('synapse', self.syn_class)]))

    def initialization(self,activation_func, coding_rule,act_init =(-75,-4), parameters = np.array([0.02,0.2,-65,6])):
        for n_id in range (self.r_size):
            new_neuron = self.neu_class(id = n_id, activation_func =activation_func, coding_rule= coding_rule)
            self.neuron_list = np.concatenate((self.neuron_list,[new_neuron]),axis= 0)
        for s_id in range (self.s_number):
            self.__init_connect(s_id)


    def neu_initialization(self):
        for neuron in self.neuron_list:
            neuron.initialization()


    def connect(self,s_id,pre_neuron,post_neuron,delay,weight):
        new_synapse = self.syn_class(id= s_id, pre_neuron = pre_neuron, post_neuron = post_neuron,
                                      delay = delay,weight = weight)
        new_synapse.register()
        self.synapse_list = np.concatenate((self.synapse_list,[new_synapse]),axis=0)


    def reset(self):
        pass


    def __init_connect(self,s_id):
        conn = np.arange(self.r_size)
        np.random.shuffle(conn)
        conn = conn[:2]
        pre_neuron = self.neuron_list[conn[0]]
        post_neuron = self.neuron_list[conn[1]]
        delay = np.arange(1,MAX_SYNAPSE_DELAY+1)
        np.random.shuffle(delay)
        delay = delay[0]
        weight = np.random.normal(0, 1)
        self.connect(s_id,pre_neuron,post_neuron,delay,weight)










