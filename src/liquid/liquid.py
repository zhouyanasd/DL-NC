'''
The liquid is only one kind of data for input,
 and no connections between different reservoirs
'''

import numpy as np
import src

from src.core import Base,MAX_OPERATION_TIME
from src.storage import Sto_state

class Liquid(Base):
    def __init__(self, data, label, input_class, res_number = 1, read_number =1):
        self.data = data
        self.label = label[2]
        self.res_number = res_number
        self.read_number = read_number
        self.reservoir_list = np.array([], dtype = np.dtype([('reservoir', src.reservoir.Reservoir)]))

        self.input_class = getattr(src.input,input_class)
        self.input_list = np.array([], dtype = np.dtype([('input', src.input.Input)]))

        self.readout_list = []


    def initialization(self):
        self.set_operation_off()
        self.set_global_time(0)
        for res_id in range(self.res_number):
            new_reservoir = src.reservoir.Reservoir(res_id,20, conn_type = 'conn_normal', n_type = 'SpikingNeuron', s_type ='Synapse')
            self.reservoir_list = np.concatenate((self.reservoir_list,[new_reservoir]),axis=0)
            new_input = self.input_class(input_size = self.data[0].shape[0], reservoir = self.reservoir_list[res_id])      # based on the type of pre-processed data
            self.input_list = np.concatenate((self.input_list,[new_input]), axis=0)
            # the initial sequence cannot be changed
            new_reservoir.initialization('izhikevich_spiking','rate_window')
            new_input.initialization()
            new_reservoir.neu_initialization()
        for read_id in range(self.read_number):
            new_readout = src.readout.LMS_readout(read_id)
            new_readout.add_pre_reservoir(self.reservoir_list[0])
            new_readout.add_read_neuron_s()
            new_readout.initialization('decay_exponential_window')
            self.readout_list.append(new_readout)



    def add_reservoir(self):
        pass


    def add_input(self):
        pass


    def conn_input_reservoir(self):
        pass


    def add_readout(self,new_readout):
        self.readout_list.append(new_readout)
        self.read_number += 1


    def conn_readout(self):
        pass


    # TODO: the group may be set as loops
    def operate(self, group):
        while (self.get_operation()):
            t = self.get_global_time()
            for i in self.input_list:
                try:                                                                                                    # input_t will be set to zeros when beyond the index of the data
                    i.input_t = self.data[group][:,t]
                except IndexError:
                    feature = self.data[group].shape[0]
                    i.input_t = np.zeros(feature)
            self.advance()
            self.add_global_time(1)
            self.show_operation_time(10)
            if self.get_global_time() > MAX_OPERATION_TIME:
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


    def reset_test(self):
        self.liquid_stop()
        self.set_global_time(0)
        for res in self.reservoir_list:
            res.reset_test()
        for readout in self.readout_list:
            readout.reset_test()

    def pre_train_res(self):
        self.liquid_start()
        self.operate(2)
        # sto = Sto_state()
        for readout in self.readout_list:
            readout.get_state_all()
            readout.code_state_all()
            # sto.save_state(readout.pre_state,readout.id)
        self.liquid_stop()

    def train_readout(self):
        for readout in self.readout_list:
            readout.LMS_train(self.label)


    def test(self, test_data):
        self.data = test_data
        self.liquid_start()
        self.operate(0)
        output_list = []
        for readout in self.readout_list:
            readout.get_state_all()
            readout.code_state_all()
            output = readout.LMS_test()
            output_list.append(output)
        return output_list



