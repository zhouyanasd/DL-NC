import src
import numpy as np
import matplotlib.pyplot as plt


#-----------------simulation setting----------------
# define and generate the data
np.random.seed(107)
Data, cla= src.data.Simple(1,src.core.MAX_OPERATION_TIME,3).Tri_function_test()
# Data= src.data.Simple(1,1000,3).Possion()
# define and initialize the liquid
Liquid = src.liquid.Liquid(Data, cla,'Input', 1)
Liquid.initialization()
Liquid.liquid_start()
Liquid.operate(2)


#--------------------plot result---------------------
vis = src.vis.Visualization(src.core.Base().get_global_time())
mem = Liquid.reservoir_list[0].neuron_list[2].membrane_potential
# # print(mem)
I = Liquid.reservoir_list[0].neuron_list[2].I
t = np.arange(0,mem.shape[0])
fig = plt.figure()
plt.plot(t[:],mem[:,0])
fig2 = plt.figure()
plt.plot(t[:],mem[:,1])
fig3 = plt.figure()
plt.plot(I)
fig5 = plt.figure()
plt.plot(Data[2][0])
plt.show()

fig4 = plt.figure()
vis.add_fired_fig(fig4,Liquid)
vis.show()

