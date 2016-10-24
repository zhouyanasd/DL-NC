from scipy.integrate import odeint
import numpy as np
import theano
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

def Izhikevich(w,t,ii,j):
    a,b,c,d =0.02,0.2,-65,6
    # I = -500*(np.sin(t)+1)
    v,u =w
    # print("I:", I)
    # print("i:", ii(t))
    temp = np.array([0.04*(v**2)+5*v+140-u+ii(t),a*(b*v-u)])
    if temp[0] >=30:
        v=c
        u=u+d
        print(t,":",temp[0])
    return temp


t = np.arange(0,100,0.1)
def ii(t):
    i = 10
    return i

def h_Izh():
    W = theano.shared()
    track3 = odeint(Izhikevich,(-75,-4),t,(ii,1))
    print("hehe")
    return track3
# track = odeint(Leaky_IandF,(0,0),t,(1,0.2,1))
# track2 = odeint(quadratic,(-0.19,0),t,(1.5,1,-0.4,-0.2))
# track3 = odeint(Izhikevich,(-16,-56),t,(ii,1))
track3 = h_Izh()

fig = plt.figure()
# plt.plot(t,track3)
plt.plot(t[250:270],track3[250:270,0])
# print(track2[0:22,0])
fig2 = plt.figure()
plt.plot(t[250:270],track3[250:270,1])
plt.show()
