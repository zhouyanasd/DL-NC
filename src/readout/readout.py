import numpy as np
import src

from ..core import Base

class Readout(Base):
    def __init__(self):
        self.pre_reservoir_list = []
        self.read_number = 0

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
        pass

    def output(self):
        pass