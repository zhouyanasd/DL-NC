import numpy as np

from . import Coding
from ..core import Base, IZK_INTER_SCALE
from ..function import ActFunction
from ..synapse import Synapse
from ..input import Input


class SpikingNeuron(Base):
    def __init__(self, id, activation_func, coding_rule, act_init =(-75,-4), act_pram = np.array([0.02,0.2,-65,6])):
        self.id = id                                                                                                    # the neuron id

        self.fired = False                                                                                              # is fired at last time slot
        self.fired_sequence = np.array([])                                                                              # neuron fired sequence for all the time slot
        self.membrane_potential = np.array([]).reshape(0,2)                                                             # membrane potential sequence
        self.membrane_potential_now = np.zeros((1,2))                                                                   # the membrane potential for now
        self.I = np.array([])                                                                                           # the transformed input sequence
        self.I_now = 0                                                                                                  # the transformed input for now
        self.pre_synapse = np.array([], dtype =np.dtype([('synapse', Synapse)]))                                        # the pre_synapse array
        self.post_synapse = np.array([], dtype =np.dtype([('synapse', Synapse)]))                                       # the pose_synapse array
        self.input = np.array([], dtype =np.dtype([('input', Input),('index',np.int64)]))                               # the external input list

        self.activation_func = getattr(ActFunction(),activation_func)                                                   # activation function for this neuron
        self.coding = getattr(Coding(self.in_size),coding_rule)                                                         # coding rule for this neuron
        self.__init = act_init
        self.__act_pram = act_pram                                                                                      # pram = {'inti':(-75,-4),'a':0.02,'b':0.2,'c':-65,'d':6}
        self.in_size = 0                                                                                                # the input size

        self.coming_fired = np.array([],dtype= np.dtype([('time', np.int64),('coming_fired_sequence',np.array)]))

    def __get_input(self):
        pass

    def __trans_input(self,input_t):
        W = self.__get_weight()
        l = self.coding(input_t)
        self.I_now = np.dot(W,l[:, np.newaxis]).reshape(1,)*IZK_INTER_SCALE
        self.I = np.hstack((self.I,self.I_now))


    # the input and weight will be calculate by synapse and input
    def __get_weight(self):
        W = np.zeros((1,self.in_size))
        for i in range(self.pre_synapse.size):
            W[0,i] = self.pre_synapse[i].weight
        for j in range(self.input.size):
            input_index = self.input[j]['index']                                                                        # get the input index
            W[0,self.pre_synapse.size+j] = self.input[j]['input'].conn_neuron[input_index,self.id]['weight']            # get the weight from the input to this neuron
        return W


    # this function must be call by reservoir in initialization
    # the input size only can be confirmed after all the synapses resigned
    def initialization(self):
        self.in_size = np.size(self.pre_synapse)+np.size(self.input)


    def activate(self,input_t):
        self.__trans_input(input_t)
        a,b,c,d = self.__act_pram[0] ,self.__act_pram[1],self.__act_pram[2],self.__act_pram[3]
        self.membrane_potential_now= self.activation_func(self.I_now,self.__init,a,b,c,d)
        if self.membrane_potential_now[1,0]<30:
           self.fired = False
           self.__init = (self.membrane_potential_now[1,0],self.membrane_potential_now[1,1])
        else:
           self.fired = True
           self.fired_sequence = np.concatenate((self.fired_sequence,[self.get_global_time()]),axis=0)
           self.__init = (c,self.membrane_potential_now[1,1]+d)
        self.membrane_potential = np.vstack((self.membrane_potential, self.membrane_potential_now[1]))

    #TODO: the neuron or synapse need the method to transmit or receive the spiking

    def receive(self):
        pass

