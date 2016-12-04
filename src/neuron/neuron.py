import numpy as np

from ..core import Base
from . import Coding
from ..function import ActFunction


class SpikingNeuron(Base):
    def __init__(self, id, in_size, activation_func, coding_rule, init = (-75,-4)):
        self.id = id

        self.fired = False                                                          # is fired at last time slot
        self.fired_sequence = np.array([])                                          # neuron fired sequence for all the time slot
        self.membrane_potential = np.array([]).reshape(0,2)                         # membrane potential sequence
        self.membrane_potential_now = np.zeros((1,2))                               # the membrane potential for now
        self.I = np.array([])                                                       # the transformed input sequence
        self.I_now = 0                                                              # the transformed input for now
        self.I_inject = 0                                                           # the transformed inject input for now
        self.activation_func = getattr(ActFunction(),activation_func)               # activation function for this neuron
        self.coding = getattr(Coding(in_size),coding_rule)                          # coding rule for this neuron

        #TODO: the input size will be get from the synapses connected to this neuron
        self.in_size = in_size

        self.__init = init

    def __trans_input(self,input_t):

        #TODO: the input and weight will be calculate by synapse
        W = np.abs(np.random.normal(0, 1, (1,self.in_size)))

        l=self.coding(input_t)
        self.I_now=(np.dot(W,l[:, np.newaxis])+self.b).reshape(1,)*20
        self.I = np.hstack((self.I,self.I_now))



    def activate(self,input_t,a=0.02,b=0.2,c=-65,d=6):

        self.__trans_input(input_t)

        self.membrane_potential_now= self.activation_func(self.I_now,self.__init,a,b,c,d)
        if self.membrane_potential_now[1,0]<30:
           self.fired = False
           self.__init = (self.membrane_potential_now[1,0],self.membrane_potential_now[1,1])
        else:
           self.fired = True
           self.__init = (c,self.membrane_potential_now[1,1]+d)
        self.membrane_potential = np.vstack((self.membrane_potential ,self.membrane_potential_now[1]))
