from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(X,useStatus):
#     if useStatus:
#         return 1.0 / (1 + np.exp(-float(X)))
#     else:
#         return float(X)

def Leaky_IandF(u,t,tor,r,i):
    x, y =u
    sin = np.sin(50*t)*0.1+np.sin(30*t)+np.sin(100*t)+i
    return np.array([(-x+r*i)/tor,(-y+r*sin)/tor])

def quadratic(u,t,tor,a,ur,uc):
    x, y =u
    func = a*(x-ur)*(x-uc)
    return np.array([func/tor,y])


t = np.arange(0,35,1)
track = odeint(Leaky_IandF,(0,0),t,(1,0.2,1))
track2 = odeint(quadratic,(-0.19,0),t,(1.5,1,-0.4,-0.2))

fig = plt.figure()
# plt.plot(t,track)
plt.plot(t[0:23],track2[0:23,0])
print(track2[0:22,0])
plt.show()

class Spiking_neuron(object):
    def __init__(self):
        self.W=1