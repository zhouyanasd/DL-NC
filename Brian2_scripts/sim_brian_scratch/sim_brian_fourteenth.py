from brian2 import *
from scipy.optimize import leastsq
import scipy as sp
from sklearn.preprocessing import MinMaxScaler

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


def classification(thea, data):
    data_n = normalization_min_max(data)
    data_class = []
    for a in data_n:
        if a >= thea:
            b = 1
        else:
            b = 0
        data_class.append(b)
    return np.asarray(data_class), data_n


def normalization_min_max(arr):
    arr_n = arr
    for i in range(arr.size):
        x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr_n[i] = x
    return arr_n


def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))


def load_Data_JV(path="../Data/jv/train.txt"):
    data = np.loadtxt(path, delimiter=None)
    s = open(path, 'r')
    i = -1
    size_d = []
    while True:
        lines = s.readline()
        i += 1
        if not lines:
            break
        if lines == '\n':  # "\n" needed to be added at the end of the file
            i -= 1
            size_d.append(i)
            continue
    return MinMaxScaler().fit_transform(data), size_d


def get_label(obj, t, size_d, path="../Data/jv/size.txt"):
    if t == "train":
        data_l = np.loadtxt(path, delimiter=None).astype(int)[1]
    elif t == "test":
        data_l = np.loadtxt(path, delimiter=None).astype(int)[0]
    else:
        raise TypeError("t must be 'train' or 'test'")
    data_l = np.cumsum(data_l)
    data_l_ = np.concatenate(([0], data_l))
    size_d_ = np.asarray(size_d) + 1
    size_d_ = np.concatenate(([0], size_d_))
    label = []
    for i in range(len(data_l_) - 1):
        for j in range(data_l_[i], data_l_[i + 1]):
            for l in range(size_d_[j], size_d_[j + 1]):
                if i == obj:
                    label.append(1)
                else:
                    label.append(0)
    return asarray(label)


# -----parameter setting-------
data, size_d = load_Data_JV()
label = get_label(2, "train", size_d)
n = 30
duration = len(data) * defaultclock.dt
threshold = 0.5

equ_in = '''
I = stimulus(t,i) : 1
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

stimulus = TimedArray(data, dt=defaultclock.dt)

Input = NeuronGroup(len(data.T), equ_in, method='linear')
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
S4.w = '0'

# ------run----------------
m1 = StateMonitor(Input, ('I'), record=True)
m3 = StateMonitor(G_readout, ('I'), record=True)
m4 = StateMonitor(G, ('I'), record=True)

run(duration)

# ----lms_readout----#
Data, para = readout(m3.I, label)
print(para)
label_t = lms_test(Data, para)
label_t_class, data_n = classification(threshold, label_t)

# ------vis----------------
fig0 = plt.figure(figsize=(20, 10))
subplot(511)
plot(m1.t / ms, m1.I[0], '-b', label='I')
subplot(512)
plot(m1.t / ms, m1.I[1], '-b', label='I')
subplot(513)
plot(m1.t / ms, m1.I[2], '-b', label='I')
subplot(514)
plot(m1.t / ms, m1.I[3], '-b', label='I')
subplot(515)
plot(m1.t / ms, m1.I[4], '-b', label='I')

fig1 = plt.figure(figsize=(20, 4))
subplot(111)
plt.scatter(m3.t / ms, label_t_class, s=2, color="red", marker='o', alpha=0.6)
plt.scatter(m3.t / ms, label, s=3, color="blue", marker='*', alpha=0.4)
plt.plot(m3.t / ms, data_n, color="green")
axhline(threshold, ls='--', c='r', lw=1)

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
