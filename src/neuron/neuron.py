import numpy as np
import src

from ..core import Base, IZK_INTER_SCALE
from ..function import Coding
from ..function import ActFunction


class SpikingNeuron(Base):
    def __init__(self, id, activation_func, coding_rule, act_init =(-75,-4), parameters = np.array([0.02,0.2,-65,6]),
                 *args, **kwargs):
        self.id = id                                                                                                    # the neuron id
        self.fired = False                                                                                              # is fired at last time slot
        self.fired_sequence = np.array([])                                                                              # neuron fired sequence for all the time slot
        self.membrane_potential = np.array([]).reshape(0,2)                                                             # membrane potential sequence
        self.membrane_potential_now = np.zeros((1,2))                                                                   # the membrane potential for now
        self.I = np.array([])                                                                                           # the transformed input sequence
        self.I_now = 0                                                                                                  # the transformed input for now
        self.pre_synapse = np.array([], dtype =np.dtype([('synapse', src.synapse.Synapse)]))                            # the pre_synapse array
        self.post_synapse = np.array([], dtype =np.dtype([('synapse', src.synapse.Synapse)]))                           # the pose_synapse array
        self.input = np.array([], dtype =np.dtype([('input', src.input.Input),('index',np.int64)]))                     # the external input list
        self.coming_fired = 0
        self.in_size = 0                                                                                                # the input size
        self.__func = activation_func                                                                                   # activation function for this neuron
        self.__coding = coding_rule                                                                                     # coding rule for this neuron
        self.__init = act_init
        self.__parameters = parameters                                                                                  # pram = {'inti':(-75,-4),'a':0.02,'b':0.2,'c':-65,'d':6}

    # this function must be call by reservoir in neu_initialization
    # the input size only can be confirmed after all the self.synapses and self.input resigned
    def initialization(self):
        self.in_size = np.size(self.pre_synapse)+ np.size(self.input)
        self.coming_fired = np.array([]).reshape(self.pre_synapse.size,0)
        self.activation_func = getattr(ActFunction(), self.__func)                                                      # activation function for this neuron
        self.coding = getattr(Coding(self.in_size), self.__coding)



    def activate(self):
        self._trans_input()
        p = self.__parameters
        self.membrane_potential_now = self.activation_func(self.I_now,self.__init,p[0],p[1],p[2],p[3])
        if self.membrane_potential_now[1,0] < 30:
           self.fired = False
           self._trans_fired()
           self.__init = (self.membrane_potential_now[1,0],self.membrane_potential_now[1,1])
        else:
           self.fired = True
           self.membrane_potential_now[1,0] = 30
           self._trans_fired()
           self.fired_sequence = np.concatenate((self.fired_sequence,[self.get_global_time()]),axis=0)
           self.__init = (p[2],self.membrane_potential_now[1,1]+p[3])
        self.membrane_potential = np.vstack((self.membrane_potential, self.membrane_potential_now[1]))


    def receive_spiking(self):
        coming = np.array([]).reshape(0,1)
        for syn in self.pre_synapse:
            coming = np.concatenate((coming,[[syn.spiking_buffer[0]]]),axis=0)
        self.coming_fired = np.concatenate((self.coming_fired,coming),axis=1)


    def reset(self):
        pass


    def _trans_fired(self):
        for i in self.post_synapse:
            i.trans_fired()


    def _get_input(self):
        spiking = self.coming_fired[:,self.get_global_time()][:,np.newaxis]
        input = np.array([]).reshape(0,1)
        for i in self.input:
            value = i['input'].input_t[i['index']]
            input = np.concatenate ((input,[[value]]),axis= 0)
        input_t = np.concatenate((spiking,input),axis= 0)
        return input_t


    # the input and weight will be calculate by synapse and input
    def _get_weight(self):
        W = np.zeros((1,self.in_size))
        for i in range(self.pre_synapse.size):
            W[0,i] = self.pre_synapse[i].weight
        for j in range(self.input.size):
            input_index = self.input[j]['index']                                                                        # get the input index
            W[0,self.pre_synapse.size+j] = self.input[j]['input'].conn_neuron[input_index,self.id]['weight']            # get the weight from the input to this neuron
        return W


    def _trans_input(self):
        input_t = self._get_input()
        W = self._get_weight()
        l = self.coding(input_t)
        self.I_now = np.dot(W,l[:, np.newaxis])[0,0]*IZK_INTER_SCALE
        self.I = np.hstack((self.I,self.I_now))