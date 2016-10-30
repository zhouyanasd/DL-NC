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
    def __init__(self, in_size, activation_func,d_t = 5):
        self.in_size = in_size # the input size (the number of persynaptic)
        self.d_t = d_t  # time window
        self.fired = False# is fired at last time slot
        self.W = np.abs(np.random.normal(0, 1, (1,self.in_size)))
        self.b = 0.1
        self.output = 0
        self.activation_func = activation_func

        self.is_input = True #total_time_slot and is_input are operation control parameters
        self.I = np.array([]) # the transformed input(private)
        self.in_time_slot = -1 # count for the input time slot


    def input_trans(self, input,total_time_slot = 1000): #transmit the input to analog signal(private)
        time_window_buffer = np.zeros((self.in_size,self.d_t)) #tiem window buffer
        while (self.is_input == True and self.in_time_slot+1<total_time_slot and
               self.in_time_slot+1<input.shape[1]):
            self.in_time_slot = self.in_time_slot +1

            #window slide
            time_window_buffer[:,0:self.d_t-1]=time_window_buffer[:,1:self.d_t]
            time_window_buffer[:,self.d_t-1]=input[:,self.in_time_slot]

            l = np.sum(time_window_buffer,axis= 1)/self.d_t
            I= (np.dot(self.W,l[:, np.newaxis])+self.b).reshape(1,)
            self.I = np.hstack((self.I,I))


    def activate(self,input,total_time_slot = 1000):
         while (self.is_input == True and self.in_time_slot+1<total_time_slot and
               self.in_time_slot+1<input.shape[1]):
             self.in_time_slot = self.in_time_slot +1




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

# inputs = np.random.randint(size = (6,1000),low = 0, high= 2)
# neuron = SpikingNeuron(6,fun.izhikevich_spiking)
# neuron.input_trans(inputs)
# print(neuron.I.shape,np.average(neuron.I))
# fig3 = plt.figure()
# plt.plot(neuron.I)
# plt.show()
#-----------------------------------
