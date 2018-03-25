import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0,2*np.pi,0.01)
y1=np.cos(x)
y2=np.cos(x+np.pi)
y3=np.cos(x+np.pi/2)
y4=np.cos(x-np.pi/2)

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y4)
plt.show()