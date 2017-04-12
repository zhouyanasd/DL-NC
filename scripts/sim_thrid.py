import src
import numpy as np
import matplotlib.pyplot as plt

#-----------------simulation setting----------------
np.random.seed(107)
Data= src.data.Simple(1,src.core.MAX_OPERATION_TIME,3).Sin()

print(Data[2][0:1000])
#--------------------SNN topology--------------------
Liquid = src.liquid.Liquid(Data, cla_test,'Input', 1, 1)
Liquid.initialization()

#-----------------training and testing---------------


#--------------------plot result---------------------
vis = src.vis.Visualization(src.core.Base().get_global_time())

fig1 = plt.figure(figsize=(15,8))
vis.add_data_fig(fig1,Data[2][0:1000])
vis.show()


