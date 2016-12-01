import numpy as np

from ..core import Base
from . import coding


class SpikingNeuron(Base):
    def __init__(self, in_size, activation_func):
        self.id = 0

        self.fired = False      # is fired at last time slot
        self.activation_func = activation_func
        self.membrane_potential = 0

        self.in_size = in_size
        self.b = 0.1

        self.I = np.array([])   # the transformed input(private)





    def activate(self,input,init = (-75,-4),a=0.02,b=0.2,c=-65,d=6
                   ,total_time_slot = 1000):

         v = init
         temp_track = np.array([]).reshape(0,2)
         W = np.abs(np.random.normal(0, 1, (1,self.in_size)))



         self.in_time_slot = self.in_time_slot +1
         l=coding.rate_window(self.in_size,input[:,self.in_time_slot])
         I=(np.dot(W,l[:, np.newaxis])+self.b).reshape(1,)*20
         self.I = np.hstack((self.I,I))

         temp_track_t= self.activation_func(self.I[self.in_time_slot],self.in_time_slot,v,a,b,c,d)
         if temp_track_t[1,0]<30:
            v = (temp_track_t[1,0],temp_track_t[1,1])
         else:
            v = (c,temp_track_t[1,1]+d)
         temp_track = np.vstack((temp_track ,temp_track_t[1]))
         return temp_track