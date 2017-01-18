import numpy as np
import src

from ..core import Base, READOUT_TIME_WINDOW
from ..function import Coding

class Readout(Base):
    def __init__(self, coding_rule):
        self.pre_reservoir_list = []
        self.read_number = 0
        self.pre_state = np.array([])
        self.coding = getattr(Coding(1, READOUT_TIME_WINDOW), coding_rule)

    def add_pre_reservoir(self, reservoir):
        self.pre_reservoir_list.append(reservoir)

    def initialization(self):
        pass

    def add_read_neuron_s(self):
        self.read_number += 1

    def add_read_neuron_n(self, n_number):
        self.read_number += n_number

    def connect(self):
        pass

    def get_state(self):
        for res in self.pre_reservoir_list:
            for neu in res.neuron_list:
                t = Base.get_global_time()


    def output(self):
        pass