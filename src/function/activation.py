from scipy.integrate import odeint
import numpy as np

from ..core import Base

class IzhActFunc(Base):

    def __init__(self, time_scale=0.1):
        self.time_scale = time_scale


    def __izhikevich(self,w, t, I, a, b):
        v, u = w
        temp = np.array([0.04 * (v ** 2) + 5 * v + 140 - u + I, a * (b * v - u)])
        return temp


    def izhikevich_spiking(self,I,time_slot,w = (-75,-4),a=0.02,b=0.2,c=-65,d=6):
        t = np.arange(self.time_scale*time_slot,self.time_scale*(time_slot+2),self.time_scale)
        temp_track = odeint(self.__izhikevich,(w),t,(I,a,b))
        return temp_track