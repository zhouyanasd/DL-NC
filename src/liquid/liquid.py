'''
The liquid is only one kind of data for input,
 and no connections between different reservoirs
'''

import numpy as np
import src

from ..core import Base,MAX_OPERATION_TIME

class Liquid(Base):
    def __init__(self, data, input_class, r_number = 1):
        self.data = data
        self.r_number = r_number
        self.reservoir_list = np.array([], dtype =np.dtype([('reservoir', src.reservoir.Reservoir)]))

        self.input_class =getattr(src.input,input_class)
        self.input_list = np.array([], dtype =np.dtype([('input', src.input.Input)]))


    def initialization(self):
        self.set_operation_off()
        self.set_global_time(0)
        for r_id in range(self.r_number):
            new_reservoir = src.reservoir.Reservoir(r_id,20)
            self.reservoir_list = np.concatenate((self.reservoir_list,[new_reservoir]),axis=0)
            new_input = self.input_class(input_size = self.data[0].shape[0],reservoir = self.reservoir_list[r_id])      # based on the type of pre-processed data
            self.input_list = np.concatenate((self.input_list,[new_input]),axis=0)
            # the initial sequence cannot be changed
            new_reservoir.initialization('izhikevich_spiking','rate_window')
            new_input.initialization()
            new_reservoir.neu_initialization()


    def add_reservoir(self):
        pass


    def add_input(self):
        pass


    def conn_input_reservoir(self):
        pass


    def add_readout(self):
        pass


    def conn_readout(self):
        pass


    # TODO: the group may be set as loops
    def operate(self, group):
        while (self.get_operation()):
            t = self.get_global_time()
            print(t)
            for i in self.input_list:
                try:                                                                                                    # input_t will be set to zeros when beyond the index of the data
                    i.input_t = self.data[group][:,t]
                except IndexError:
                    feature = self.data[group].shape[0]
                    i.input_t = np.zeros(feature)
            self.advance()
            self.add_global_time(1)
            # print("t: ", self.get_global_time())
            if self.get_global_time()>MAX_OPERATION_TIME :
                self.set_operation_off()


    def advance(self):
        for res in self.reservoir_list:
            for neu in res.neuron_list:
                neu.receive_spiking()
                neu.activate()
            for syn in res.synapse_list:
                syn.adjust_weight()
                syn.trans_fired()


    def liquid_start(self):
        self.set_operation_on()


    def liquid_stop(self):
        self.set_operation_off()


    def reset(self):
        pass


    def input_group_flow(self):
        self.reset()