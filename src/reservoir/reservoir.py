import numpy as np

from ..core import Base

class SpikingNeuron(Base):
    def __init__(self, in_size, activation_func,d_t = 5):
        #public
        self.in_size = in_size  # the input size (the number of persynaptic)
        self.W = np.abs(np.random.normal(0, 1, (1,self.in_size)))
        self.b = 0.1
        self.output = 0
        self.in_time_slot = -1  # count for the input time slot

        self.is_input = True    # total_time_slot and is_input are operation control parameters
        self.fired = False      # is fired at last time slot
        self.activation_func = activation_func
        self.d_t = d_t          # time window


        #private
        self.I = np.array([])   # the transformed input(private)
        self.__time_window_buffer = np.zeros((self.in_size,self.d_t)) #tiem window buffer

    def __input_trans(self, input): #transmit the input to analog signal(private)
        #window slide
        self.__time_window_buffer[:,0:self.d_t-1]=self.__time_window_buffer[:,1:self.d_t]
        self.__time_window_buffer[:,self.d_t-1]=input
        #caculate the value
        l = np.sum(self.__time_window_buffer,axis= 1)/self.d_t
        I= (np.dot(self.W,l[:, np.newaxis])+self.b).reshape(1,)*20
        self.I = np.hstack((self.I,I))


    def activate(self,input,w0 = (-75,-4),a=0.02,b=0.2,c=-65,d=6
                   ,track=np.array([]).reshape(0,2),total_time_slot = 1000):
         w = w0
         temp_track = track
         while (self.is_input == True and self.in_time_slot+1<total_time_slot and
               self.in_time_slot+1<input.shape[1]):
            self.in_time_slot = self.in_time_slot +1
            self.__input_trans(input[:,self.in_time_slot])

            temp_track_t= self.activation_func(self.I[self.in_time_slot],self.in_time_slot,w,a,b,c,d)
            if temp_track_t[1,0]<30:
                w = (temp_track_t[1,0],temp_track_t[1,1])
            else:
                w = (c,temp_track_t[1,1]+d)
            temp_track = np.vstack((temp_track ,temp_track_t[1]))
         return temp_track