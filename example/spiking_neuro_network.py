from scipy.integrate import odeint
import numpy as np

class inter_act_func(object):
    def __init__(self, t):
        self.w=1
        self.t = np.arange(0,100,100/t)
        self.I = -500


    def Leaky_IandF(self,u,t,tor,r,i):
        x, y =u
        sin = np.sin(50*t)*0.1+np.sin(30*t)+np.sin(100*t)+i
        return np.array([(-x+r*i)/tor,(-y+r*sin)/tor])

    def quadratic(self,u,t,tor,a,ur,uc):
        x, y =u
        func = a*(x-ur)*(x-uc)
        return np.array([func/tor,y])

    def Izhikevich(self,w,t):
        a,b,c,d =0.02,0.2,-65,8
        I = -500*(np.sin(t)+1)
        u,v =w
        return np.array([0.04*(v**2)+5*v+140-u+I,a*(b*v-u)])









class spiking_neuron(object):
    def __init__(self, input, activation_func=inter_act_func.Izhikevich):
        self.W=1
        self.b=0.1
        self.t = 1
        self.output = 0
        self.activate_func=activation_func

    def activate(self):
        self.output = odeint(self.activation_func,self.t)