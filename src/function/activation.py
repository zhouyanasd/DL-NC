from scipy.integrate import odeint
import numpy as np

from ..core import Base
from ..core import TIME_SCALE as time_scale

class ActFunction(Base):

    def __init__(self):
        pass

    def __izhikevich(self,w, t, I, a, b):
        v, u = w
        temp = np.array([0.04 * (v ** 2) + 5 * v + 140 - u + I, a * (b * v - u)])
        return temp


    def izhikevich_spiking(self,I,w = (-75,-4),a=0.02,b=0.2,c=-65,d=6):
        t = np.arange(time_scale*Base().get_global_time(),time_scale*(Base().get_global_time()+2),time_scale)
        membrane_potential_temp = odeint(self.__izhikevich,(w),t,(I,a,b))
        return membrane_potential_temp