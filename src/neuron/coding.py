import numpy as np
from ..core import INPUT_TIME_WINDOW as d_t

def rate_window(in_size, input): #transmit the input to analog signal(private)
        time_window_buffer = np.zeros((in_size,d_t)) #tiem window buffer
        #window slide
        time_window_buffer[:,0:d_t-1]=time_window_buffer[:,1:d_t]
        time_window_buffer[:,d_t-1]=input
        #caculate the value
        l = np.sum(time_window_buffer,axis= 1)/d_t
        return l