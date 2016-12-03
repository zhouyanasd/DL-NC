import numpy as np
from ..core import INPUT_TIME_WINDOW as d_t
from ..core import Base

class Coding(Base):

    def __init__(self,in_size,single_input):
        self.time_window_buffer = np.zeros((in_size,d_t)) #tiem window buffer
        self.in_size = in_size
        self.__input = single_input()

    def rate_window(self): #transmit the input to analog signal(private)
        #window slide
        self.time_window_buffer[:,0:d_t-1]=self.time_window_buffer[:,1:d_t]
        self.time_window_buffer[:,d_t-1]=self.__input
        #caculate the value
        l = np.sum(self.time_window_buffer,axis= 1)/d_t
        return l

    def rate_simple(self):
        pass

    def rate_Gaussian(self):
        pass

    def spike_simple(self):
        pass