import numpy as np
import src

from src.core import Base

class Liquid2(Base):
    def __init__(self,data = 0,input_class = 'Input',res_number = 1, read_number =1):
        self.data = data
        self.res_number = res_number
        self.read_number = read_number
        self.input_class = getattr(src.input,input_class)
        self.input_list = np.array([], dtype = np.dtype([('input', src.input.Input)]))
        self.reservoir_list = np.array([], dtype = np.dtype([('reservoir', src.reservoir.Reservoir)]))
