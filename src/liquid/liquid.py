'''
The liquid is only one kind of data for input,
 and no connections between different reservoirs
'''

import numpy as np
import src

from ..core import Base

class Liquid(Base):
    def __init__(self, data, input_class, r_number = 1):
        self.data = data
        self.r_number = r_number
        self.reservoir_list = np.array([], dtype =np.dtype([('reservoir', src.reservoir.Reservoir)]))

        self.input_class =getattr(src.input,input_class)
        self.input = np.array([], dtype =np.dtype([('input', src.input.Input)]))


    def initialization(self):
        self.set_global_time(0)
        for r_id in range(self.r_number):
            new_reservoir = src.reservoir.Reservoir(r_id,10,5)
            new_reservoir.initialization('izhikevich_spiking','rate_window')
            self.reservoir_list = np.concatenate((self.reservoir_list,[new_reservoir]),axis=0)
            new_input = self.input_class(input_size = self.data[0].shape[0],reservoir = self.reservoir_list[r_id])      # based on the type of pre-processed data
            new_input.initialization()
            self.input = np.concatenate((self.input,[new_input]),axis=0)


    def liquid_start(self):
        t = self.get_global_connection()



    def time_flow(self):
        pass


    def reset(self):
        pass


    def advance(self):
        pass