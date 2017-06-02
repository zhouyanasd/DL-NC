from brian2 import *
from scipy.optimize import leastsq
import scipy as sp

start_scope()
np.random.seed(100)

#------define function------------
def lms_train(p0,Zi,Data):
    def error(p, y, args):
        l = len(p)
        f = p[l - 1]
        for i in range(len(args)):
            f += p[i] * args[i]
        return f - y
    Para = leastsq(error,p0,args=(Zi,Data))
    return Para[0]

def lms_test(Data, p):
    l = len(p)
    f = p[l - 1]
    for i in range(len(Data)):
        f += p[i] * Data[i]
    return f

def readout(M,Z):
    n = len(M)
    Data=[]
    for i in M:
        Data.append(i[1:])
    p0 = [1]*n
    p0.append(0.1)
    para = lms_train(p0, Z, Data)
    return Data,para

def classification(thea, data):
    def normalization_min_max(arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr))/(np.max(arr)- np.min(arr))
            arr_n[i] = x
        return arr_n
    data_n = normalization_min_max(data)
    data_class = []
    for a in data_n:
        if a >=thea:
            b = 1
        else:
            b = 0
        data_class.append(b)
    return np.asarray(data_class),data_n

def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def load_Data_JV(path = "Data/jv/train.txt"):
    data = np.loadtxt(path, delimiter=None)
    s = open(path, 'r')
    i = -1
    size_d = []
    while True:
        lines = s.readline()
        i += 1
        if not lines:
            break
        if lines == '\n':#"\n" needed to be added at the end of the file
            i -= 1
            size_d.append(i)
            continue
    return data,size_d

def get_label(obj,t,size_d,path = "Data/jv/size.txt"):
    if t =="train":
        data_l = np.loadtxt(path, delimiter=None).astype(int)[1]
    elif t=="test":
        data_l = np.loadtxt(path, delimiter=None).astype(int)[0]
    else:
        raise TypeError("t must be 'train' or 'test'")
    label = []
    for i in range(len(data_l)):

        for j in range(data_l[i]):
            for l in range(size_d[j]):
                if i == obj:
                    label.append(1)
                else:
                    label.append(0)
    return label



#-----parameter setting-------
n = 20
time_window = 5*ms
duration = 200 * ms
interval_l = 8
interval_s = ms
threshold = 0.5

equ = '''
dv/dt = (I-v) / (0.3*ms) : 1 (unless refractory)
dg/dt = (-g)/(0.15*ms) : 1
dh/dt = (-h)/(0.145*ms) : 1
I = (g-h)*40 +I_0: 1
I_0 = stimulus(t)*w_g:1
w_g : 1
'''

equ_1 = '''
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = (g-h)*30 : 1
'''

on_pre = '''
h+=w
g+=w
'''

#-----simulation setting-------

data , size_d= load_Data_JV()
label = get_label(1,"train",size_d)

print(label)
