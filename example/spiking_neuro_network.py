from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def ii(t):
    i = 50
    return i


class InterActFunc(object):

    def __init__(self, time_scale=0.1):
        self.time_scale = time_scale


    def __izhikevich(self,w, t, I, a, b):
        v, u = w
        temp = np.array([0.04 * (v ** 2) + 5 * v + 140 - u + I(t), a * (b * v - u)])
        return temp


    def izhikevich_spiking(self,I,w0 = (-75,-4),a=0.02,b=0.2,c=-65,d=6,
                   time = 1000,track=np.array([]).reshape(0,2)):
        w = w0
        temp_track = track
        for i in range(0,time,1):
            t = np.arange(self.time_scale*i,self.time_scale*(i+2),self.time_scale)
            temp_track_t = odeint(self.__izhikevich,(w),t,(I,a,b))
            temp_track = np.vstack((temp_track ,temp_track_t[1]))
            if temp_track_t[1,0]<30:
                w = (temp_track_t[1,0],temp_track_t[1,1])
            else:
                w = (c,temp_track_t[1,1]+d)
        return temp_track



class SpikingNeuron(object):
    def __init__(self, input, activation_func):
        self.d_t =5  # time window
        self.fired = False# is fired at last time slot
        self.W=1
        self.b=0.1
        self.t = 1
        self.output = 0
        self.activation_func=activation_func

    def input_trans(self):
        pass


    def activate(self):
        pass

#---------------test---------------
fun = InterActFunc()
track = fun.izhikevich_spiking(I=ii)
print(track[:,0],track.shape)


t = np.arange(0,100,0.1)
fig = plt.figure()
plt.plot(t[:],track[:,0])
fig2 = plt.figure()
plt.plot(t[:],track[:,1])
plt.show()
#-----------------------------------
