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

def Izhikevich(w,t,ii,a=0.02,b=0.2,c=-65,d=6):
    # a,b,c,d =0.02,0.2,-65,6
    # I = -500*(np.sin(t)+1)
    v,u =w
    # print("I:", I)
    # print("i:", ii(t))
    temp = np.array([0.04*(v**2)+5*v+140-u+ii(t),a*(b*v-u)])
    # if temp[0] >=30:
    #     v=c
    #     u=u+d
    print(t,":",temp[1])
    return temp



def ii(t):
    i = 50
    return i

def h_Izh(track=np.array([]).reshape(0,2),w0 = (-75,-4),time = 1,time_scale = 0.1):
    w =w0
    temp_track = track
    for i in range(0,time,1):
        t = np.arange(time_scale*i,time_scale*(i+2),time_scale)#i have to plus at lest 2 time slot
        temp_track_t = odeint(Izhikevich,(w),t,(ii,0.02,0.2,-65,6))
        temp_track = np.vstack((temp_track ,temp_track_t[1]))
        if temp_track_t[1,0]<30:
            w=(temp_track_t[1,0],temp_track_t[1,1])
        else:
            w = (-65,temp_track_t[1,1]+6)
    return temp_track
# track = odeint(Leaky_IandF,(0,0),t,(1,0.2,1))
# track2 = odeint(quadratic,(-0.19,0),t,(1.5,1,-0.4,-0.2))
# track3 = odeint(Izhikevich,(-16,-56),t,(ii,1))
track3 = h_Izh()
print(track3[:,0],track3.shape)


# t = np.arange(0,100,0.1)
# fig = plt.figure()
# # plt.plot(t,track3)
# plt.plot(t[:],track3[:,0])
# # print(track2[0:22,0])
# fig2 = plt.figure()
# plt.plot(t[:],track3[:,1])
# plt.show()
