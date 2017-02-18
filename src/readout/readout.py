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
        self.pre_state = np.array([])
        self.weight = 0


    def add_pre_reservoir(self, reservoir):
        self.pre_reservoir_list.append(reservoir)

    def initialization(self, coding_rule):
        neu_n =0
        for res in self.pre_reservoir_list:
            neu_n += res.neuron_list.size
        self.coding = getattr(Coding(neu_n, READOUT_TIME_WINDOW), coding_rule)
        self.pre_state = np.array([]).reshape(neu_n,0)

    def add_read_neuron_s(self):
        self.read_number += 1

    def add_read_neuron_n(self, n_number):
        self.read_number += n_number

    def connect(self):
        pass

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

    def output_t(self):
        t = self.get_global_time()
        state = self.coding(self.pre_state[t:t+READOUT_TIME_WINDOW])
        return state
