from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def ii(t):
    i = 50
    return i

class Base(object):
    __global_time = 0

    def get_global_time(self):
        return self.get_global_time

    def set_global_time(self,time):
        self.__global_time = time

    def add_global_time(self,dt):
        self.__global_time = self.__global_time + dt



class InterActFunc(Base):

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



class SpikingNeuron(Base):
    def __init__(self, in_size, activation_func,d_t = 5):
        #public
        self.in_size = in_size # the input size (the number of persynaptic)
        self.d_t = d_t  # time window
        self.fired = False# is fired at last time slot
        self.W = np.abs(np.random.normal(0, 1, (1,self.in_size)))
        self.b = 0.1
        self.output = 0
        self.activation_func = activation_func
        self.in_time_slot = -1 # count for the input time slot
        self.is_input = True #total_time_slot and is_input are operation control parameters

        #private
        self.I = np.array([]) # the transformed input(private)
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


class Synapse(Base):
    pass


#---------------test---------------
inputs = np.random.randint(size = (5,1000),low = 0, high= 2)
fun = InterActFunc()
neuron = SpikingNeuron(5,fun.izhikevich_spiking,d_t= 5)
track = neuron.activate(inputs)

print(neuron.I.shape,np.average(neuron.I))
fig3 = plt.figure()
plt.plot(neuron.I)

print(track[:,0],track.shape)
t = np.arange(0,100,0.1)
fig = plt.figure()
plt.plot(t[:],track[:,0])
fig2 = plt.figure()
plt.plot(t[:],track[:,1])

plt.show()
#-----------------------------------
