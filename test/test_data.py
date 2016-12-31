import src
import numpy as np
import matplotlib.pyplot as plt

Data, cla= src.data.Simple(1,50000,3).Tri_function()
t = np.arange(0,50000)
fig = plt.figure()
plt.plot(Data[0][0])
fig2 = plt.figure()
plt.scatter(t,cla[0])
plt.show()