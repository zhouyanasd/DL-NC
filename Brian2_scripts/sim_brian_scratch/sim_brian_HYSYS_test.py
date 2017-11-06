from brian2 import *
from brian2tools import *
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


def get_HYSYS_data(path='../../Data/HYSYS/shoulian.mat'):
    data = sio.loadmat(path)
    input_u = data['xxadu'].T[1:]
    output_y = data['xxadu'].T[0]
    temp = []
    for t in range(input_u.T.shape[0]):
        for i in range(48):
            temp.append(np.array([0] * 14))
        temp.append(input_u.T[t])
        for j in range(1):
            temp.append(np.array([0] * 14))
    input_u = np.asarray(temp).T
    return MinMaxScaler().fit_transform(input_u).T, \
           MinMaxScaler().fit_transform(output_y.reshape(-1, 1)).T[0]


def classification(thea, data):
    def normalization_min_max(arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
            arr_n[i] = x
        return arr_n

    data_n = normalization_min_max(data)
    data_class = []
    for a in data_n:
        if a >= thea:
            b = 1
        else:
            b = 0
        data_class.append(b)
    return np.asarray(data_class), data_n


def ROC(y, scores, fig_title='ROC', pos_label=1):
    def normalization_min_max(arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
            arr_n[i] = x
        return arr_n

    scores_n = normalization_min_max(scores)
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y, scores_n, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)

    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(fig_title)
    plt.legend(loc="lower right")
    return fig, roc_auc, thresholds


# -----parameter setting-------
u, y = get_HYSYS_data()
n = 10
duration = len(u) * defaultclock.dt

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

stimulus = TimedArray(u, dt=defaultclock.dt)

Input = NeuronGroup(14, equ_in, method='linear')
G = NeuronGroup(n, equ, threshold='v > 0.10', reset='v = 0', method='linear', refractory=0 * ms)
G2 = NeuronGroup(2, equ, threshold='v > 0.10', reset='v = 0', method='linear', refractory=0 * ms)
G_readout = NeuronGroup(n, equ_1, method='linear')

S = Synapses(Input, G, model, method='linear')
S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', delay=0 * ms)
S3 = Synapses(Input, G2, model, method='linear')
S4 = Synapses(G, G, 'w : 1', on_pre=on_pre, method='linear', delay=0 * ms)
S_readout = Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

# -------network S2.connect()topology----------
S.connect()
S2.connect()
S3.connect()
S4.connect(condition='i != j', p=0.1)
S_readout.connect(j='i')

S.w = 'rand()'  # '0.1+j*' + str(0.8 / n /12)
S2.w = '-rand()/2'
S3.w = '0.03+j*0.03'
S4.w = '0'

# ------run----------------
m1 = StateMonitor(Input, ('I'), record=True)
m3 = StateMonitor(G_readout, ('I'), record=True, dt=50 * defaultclock.dt)
m4 = StateMonitor(G, ('I'), record=True)

run(duration)

# ----lms_readout----#
m3.record_single_timestep()
print(m3.I[:, 1:].shape)
Data, para = readout(m3.I[:, 1:], y)
print(para)
y_t = lms_test(Data, para)
print(mse(y_t, y))

fig_roc_train, roc_auc_train, thresholds_train = ROC(y, y_t, 'ROC for train')
print('ROC of train is %s' % roc_auc_train)

# ------vis----------------
fig0 = plt.figure(figsize=(20, 4))
brian_plot(m1)

fig1 = plt.figure(figsize=(20, 8))
plt.scatter(m3.t[1:] / ms, y, s=2, color="red", marker='o', alpha=0.6)
plt.scatter(m3.t[1:] / ms, y_t, color="green")

fig2 = plt.figure(figsize=(20, 8))
brian_plot(m1)

fig3 = plt.figure(figsize=(20, 8))
brian_plot(m4)
show()