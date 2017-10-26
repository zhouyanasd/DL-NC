from brian2 import *
from scipy.optimize import leastsq
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio

prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)


# ------define function------------
def lms_train(p0, Zi, Data):
    def error(p, y, args):
        l = len(p)
        f = p[l - 1]
        for i in range(len(args)):
            f += p[i] * args[i]
        return f - y

    Para = leastsq(error, p0, args=(Zi, Data))
    return Para[0]


def lms_test(Data, p):
    l = len(p)
    f = p[l - 1]
    for i in range(len(Data)):
        f += p[i] * Data[i]
    return f


def readout(M, Z):
    n = len(M)
    Data = []
    for i in M:
        Data.append(i)
    p0 = [1] * n
    p0.append(0.1)
    para = lms_train(p0, Z, Data)
    return Data, para


def normalization_min_max(arr):
    arr_n = arr
    for i in range(arr.size):
        x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr_n[i] = x
    return arr_n


def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))


def get_WH_data(path = '../../Data/WH/WH_TestDataset.mat'):
    data = sio.loadmat(path)
    input_u = data['dataMeas'][0][0][1].T[0]
    output_y = data['dataMeas'][0][0][2].T[0]
    return MinMaxScaler().fit_transform(input_u.reshape(-1,1)).T[0], \
           MinMaxScaler().fit_transform(output_y.reshape(-1, 1)).T[0]


# -----parameter setting-------
u,y= get_WH_data()
n = 10
duration = len(u) * defaultclock.dt

equ_in = '''
I = stimulus(t) : 1
'''

equ = '''
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = (g-h)*10 + I_0 : 1
I_0 : 1 
'''

equ_1 = '''
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = (g-h)*20 : 1
'''

model = '''
w : 1
I_0_post = w * I_pre : 1 (summed)
'''

on_pre = '''
h+=w
g+=w
'''

# -----simulation setting-------

stimulus = TimedArray(u, dt=defaultclock.dt)

Input = NeuronGroup(1, equ_in, method='linear')
G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms)
G2 = NeuronGroup(2, equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms)
G_readout = NeuronGroup(n, equ_1, method='linear')

S = Synapses(Input, G, model, method='linear')
S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
S3 = Synapses(Input, G2, model, method='linear')
S4 = Synapses(G, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
S_readout = Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

# -------network S2.connect()topology----------
S.connect()
S2.connect()
S3.connect()
S4.connect(condition='i != j', p=0.1)
S_readout.connect(j='i')

S.w = 'rand()'#'0.1+j*' + str(0.8 / n /12)
S2.w = '-rand()/2'
S3.w = '0.03+j*0.03'
S4.w = 'rand()'

# ------run----------------
m1 = StateMonitor(Input, ('I'), record=True)
m3 = StateMonitor(G_readout, ('I'), record=True)
m4 = StateMonitor(G, ('I'), record=True)

run(duration)

# ----lms_readout----#
Data, para = readout(m3.I, y)
print(para)
y_t = lms_test(Data, para)
print(mse(y_t,y))

# ------vis----------------
fig0 = plt.figure(figsize=(20, 4))
plot(m1.t / ms, m1.I[0], '-b', label='I')


fig1 = plt.figure(figsize=(20, 4))
subplot(111)
plt.plot(m3.t / ms, y_t, color="red", alpha=0.6)
plt.plot(m3.t / ms, y, color="blue", alpha=0.4)

fig2 = plt.figure(figsize=(20, 8))
subplot(511)
plot(m3.t / ms, m3.I[1], '-b', label='I')
subplot(512)
plot(m3.t / ms, m3.I[3], '-b', label='I')
subplot(513)
plot(m3.t / ms, m3.I[5], '-b', label='I')
subplot(514)
plot(m3.t / ms, m3.I[7], '-b', label='I')
subplot(515)
plot(m3.t / ms, m3.I[9], '-b', label='I')

fig3 = plt.figure(figsize=(20, 8))
subplot(511)
plot(m4.t / ms, m4.I[1], '-b', label='I')
subplot(512)
plot(m4.t / ms, m4.I[3], '-b', label='I')
subplot(513)
plot(m4.t / ms, m4.I[5], '-b', label='I')
subplot(514)
plot(m4.t / ms, m4.I[7], '-b', label='I')
subplot(515)
plot(m4.t / ms, m4.I[9], '-b', label='I')
show()