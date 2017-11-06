from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import numpy as np

def get_WH_data(path = '../../Data/HYSYS/shoulian.mat'):
    data = sio.loadmat(path)
    input_u = data['xxadu'].T[1:]
    output_y = data['xxadu'].T[0]
    temp = []
    for t in range(input_u.T.shape[0]):
        for i in range (6):
            temp.append(np.array([0]*14))
        temp.append(input_u.T[t])
        for j in range (3):
            temp.append(np.array([0]*14))
    input_u = np.asarray(temp).T
    print (input_u.shape)
    return MinMaxScaler().fit_transform(input_u).T[0], \
           MinMaxScaler().fit_transform(output_y.reshape(-1,1)).T[0]


print(get_WH_data())