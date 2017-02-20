'''
This file is code by the tired stage of the framework
The readout class is coded as new rules
'''

import numpy as np
import src

from src.core import Base, READOUT_TIME_WINDOW, MAX_OPERATION_TIME
from src.function import Coding

class Readout(Base):

    def __init__(self, id):
        self.id = id
        self.pre_reservoir_list = []
        self.read_number = 0
        self.neu_n =0
        self.pre_state = np.array([])
        self.weight = 0
        self.coded_state = np.array([])


    def add_pre_reservoir(self, reservoir):
        self.pre_reservoir_list.append(reservoir)

    def initialization(self, coding_rule):
        for res in self.pre_reservoir_list:
            self.neu_n += res.neuron_list.size
        self.coding = getattr(Coding(self.neu_n, READOUT_TIME_WINDOW), coding_rule)
        self.pre_state = np.array([]).reshape(self.neu_n,0)
        self.coded_state = np.array([]).reshape(self.neu_n,0)

    def add_read_neuron_s(self):
        self.read_number += 1

    def add_read_neuron_n(self, n_number):
        self.read_number += n_number

    def reset_test(self):
        self.pre_state = np.array([]).reshape(self.neu_n,0)
        self.coded_state = np.array([]).reshape(self.neu_n,0)

    def get_state(self,t):
        t_state = np.array([]).reshape(0,1)
        for res in self.pre_reservoir_list:
            for neu in res.neuron_list:
                try:
                    if neu.fired_sequence[neu.fired_sequence == t].size !=0:
                        t_state = np.concatenate((t_state,[[1]]), axis= 0)
                    else:
                        t_state = np.concatenate((t_state,[[0]]), axis= 0)
                except IndexError:
                    t_state = np.concatenate((t_state,[[0]]), axis= 0)
        return t_state

    def get_state_t(self):
        t = self.get_global_time()
        return self.get_state(t)

    def get_state_all(self):
        t = 0
        while t < MAX_OPERATION_TIME:
            t_state = self.get_state(t)
            self.pre_state = np.concatenate((self.pre_state,t_state),axis=1)
            t += 1
            print("read_t:", t)

    def code_state_t(self,t):
        coded_state_t = self.coding(self.pre_state[:,t:t+READOUT_TIME_WINDOW], NEURON_TIME_CONSTANT = 2)
        return coded_state_t

    def code_state_all(self):
        t = 0
        while t<MAX_OPERATION_TIME:
            coded_state_t = self.code_state_t(t)[:,np.newaxis]
            self.coded_state = np.concatenate((self.coded_state,coded_state_t),axis=1)
            t += 1
