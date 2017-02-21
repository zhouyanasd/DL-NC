import src
import numpy as np
import matplotlib.pyplot as plt


#-----------------simulation setting----------------
# define and generate the data
np.random.seed(108)
Data, cla = src.data.Simple(1,src.core.MAX_OPERATION_TIME,3).Tri_function()
Data_test, cla_test = src.data.Simple(1,src.core.MAX_OPERATION_TIME,3).Tri_function_test()

# define and initialize the liquid
Liquid = src.liquid.Liquid(Data, cla,'Input', 1, 1)
Liquid.initialization()
Liquid.pre_train_res()
Liquid.train_readout()
Liquid.reset_test()
output = Liquid.test(Data_test)
print(output[0][0].size,cla_test[0][0])

#--------------------plot result---------------------
vis = src.vis.Visualization(src.core.Base().get_global_time())

fig1 = plt.figure()
vis.add_fired_fig(fig1,Liquid)
fig2 = plt.figure()
vis.add_data_fig(fig2,Data_test[0][0])
fig3 = plt.figure(figsize=(15,4))
vis.add_test_result(fig3,output[0][0],cla_test[0][0])
vis.show()