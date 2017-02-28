import src
import numpy as np
import matplotlib.pyplot as plt

#-----------------simulation setting----------------
np.random.seed(107)
Data, cla = src.data.Simple(1,src.core.MAX_OPERATION_TIME,3).Tri_function()
Data_test, cla_test = src.data.Simple(1,src.core.MAX_OPERATION_TIME,3).Tri_function_test()
print(Data_test[0][0,0:100])
print(Data_test[0][0,500:600])



#--------------------plot result---------------------