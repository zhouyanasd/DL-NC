import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# matlab文件名
matfn = '../Brian2_scripts/Data/WH/WH_TestDataset.mat'
data = sio.loadmat(matfn)

print(data)
# plt.close('all')
# xi = data['xi']
# yi = data['yi']
# ui = data['ui']
# vi = data['vi']
# plt.figure(1)
# plt.quiver(xi[::5, ::5], yi[::5, ::5], ui[::5, ::5], vi[::5, ::5])
# plt.figure(2)
# plt.contourf(xi, yi, ui)
# plt.show()
#
# sio.savemat('saveddata.mat', {'xi': xi,'yi': yi,'ui': ui,'vi': vi})