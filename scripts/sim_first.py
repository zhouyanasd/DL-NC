import src
import numpy as np
import matplotlib.pyplot as plt

#-----------------simulation setting----------------
# define and generate the data
np.random.seed(100)
Data = src.data.Simple(5,1000,3).Possion()

# define and initialize the liquid
Liquid = src.liquid.Liquid(Data, 'Input', 1)
Liquid.initialization()
Liquid.liquid_start()
Liquid.operate(2)


#--------------------plot result---------------------
mem = Liquid.reservoir_list[0].neuron_list[0].membrane_potential
print(mem)
I = Liquid.reservoir_list[0].neuron_list[0].I
t = np.arange(0,mem.shape[0])
fig = plt.figure()
plt.plot(t[:],mem[:,0])
fig2 = plt.figure()
plt.plot(t[:],mem[:,1])
fig3 = plt.figure()
plt.plot(I)


fig4 = plt.figure()
ax = fig4.add_subplot(3, 1, 2)
for j in range(Liquid.reservoir_list[0].neuron_list.size):
    fired = Liquid.reservoir_list[0].neuron_list[j].fired_sequence
    for i in fired:
        ax.scatter(i,0.5*j,alpha=.5)

plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# n = 1024
# X = np.random.normal(0,1,n)
# Y = np.random.normal(0,1,n)
# T = np.arctan2(Y,X)
#
# plt.axes([0.025,0.025,0.95,0.95])
# plt.scatter(X,Y, s=75, c=T, alpha=.5)
#
# plt.xlim(-1.5,1.5), plt.xticks([])
# plt.ylim(-1.5,1.5), plt.yticks([])
# # savefig('../figures/scatter_ex.png',dpi=48)
# plt.show()