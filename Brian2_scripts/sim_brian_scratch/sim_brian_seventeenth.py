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

def readout(M):
    print(M[0])
    n = len(M)
    Data=[]
    for i in range(n):
        x = M[i].smooth_rate(window='gaussian', width=time_window)/ Hz
        Data.append(x)
    return Data

def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def save_para(para, name):
    np.save('../Data/temp/'+str(name)+'.npy',para)

def load_para(name):
    return np.load('../Data/temp/'+str(name)+'.npy')

#-----parameter setting-------
n = 20
time_window = 10*ms
duration = 5000 * ms
duration_test = 2000*ms

taupre = taupost = 2*ms
wmax = 1
Apre = 0.2
Apost = -Apre*taupre/taupost*1.5

equ = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*20 : 1
'''

on_pre = '''
h+=w
g+=w
'''

model_STDP= '''
w : 1
dapre/dt = -apre/taupre : 1 (clock-driven)
dapost/dt = -apost/taupost : 1 (clock-driven)
'''

on_pre_STDP = '''
h+=w
g+=w
apre += Apre
w = clip(w+apost, 0, wmax)
'''

on_post_STDP= '''
apost += Apost
w = clip(w+apre, 0, wmax)
'''

