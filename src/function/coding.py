import numpy as np
from ..core import Base

class Coding(Base):

    def __init__(self,in_size, d_t):
        self.in_size = in_size
        self.d_t =d_t
        self.time_window_buffer = np.zeros((self.in_size,self.d_t))                   #tiem window buffer


    def rate_window(self,input_t):                                     #transmit the input to analog signal(private)
        #window slide
        self.time_window_buffer[:,0:self.d_t-1]=self.time_window_buffer[:,1:self.d_t]
        self.time_window_buffer[:,self.d_t-1][:, np.newaxis]=input_t
        #caculate the value
        l = np.sum(self.time_window_buffer,axis= 1)/self.d_t
        return l

    def rate_simple(self):
        pass

    def rate_Gaussian(self):
        pass

    def spike_simple(self):
        pass

    def decay_exponential(self):
        pass