import src
import numpy as np
import matplotlib.pyplot as plt


#-----------------simulation setting----------------
# define and generate the data
np.random.seed(108)
Data, cla= src.data.Simple(1,src.core.MAX_OPERATION_TIME,3).Tri_function()

# define and initialize the liquid
Liquid = src.liquid.Liquid(Data, 'Input', 1, 1)
Liquid.initialization()
Liquid.pre_train_res()
Liquid.train_readout()
Liquid.test()