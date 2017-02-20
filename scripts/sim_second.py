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
print(output[0],cla_test[0][0])