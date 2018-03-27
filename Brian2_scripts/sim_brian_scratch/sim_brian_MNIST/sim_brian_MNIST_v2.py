# ----------------------------------------
# LSM with STDP for MNIST test
# add neurons to readout layer for multi-classification(one-versus-the-rest)
# using LMS
# ----------------------------------------

from brian2 import *
from brian2tools import *
from scipy.optimize import leastsq
import scipy as sp
import struct
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)


# ------define function------------
def optimal(A, b):
    B = A.T.dot(b)
    AA = np.linalg.inv(A.T.dot(A))
    P = AA.dot(B)
    return P


def lms_test(Data, p):
    one = np.ones((Data.shape[1], 1)) #bis
    X = np.hstack((Data.T, one))
    return X.dot(p)


def readout(M, Y):
    one = np.ones((M.shape[1], 1)) #bis
    X = np.hstack((M.T, one))
    para = optimal(X, Y.T)
    return para

def normalization_min_max(arr):
    arr_n = arr
    for i in range(arr.size):
        x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr_n[i] = x
    return arr_n


def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))


def label_to_obj(label, obj):
    temp = []
    for a in label:
        if a == obj:
            temp.append(1)
        else:
            temp.append(0)
    return np.asarray(temp)


def one_versus_the_rest(label, *args, **kwargs):
    obj = []
    for i in args:
        temp = label_to_obj(label, i)
        obj.append(temp)
    try:
         for i in kwargs['selected']:
            temp = label_to_obj(label, i)
            obj.append(temp)
    except KeyError:
        pass
    return np.asarray(obj)


def trans_max_to_label(results):
    labels = []
    for result in results:
        labels.append(np.argmax(result))
    return labels


def classification(thea, data): # transform the output into class label
    data_n = normalization_min_max(data)
    data_class = []
    for a in data_n:
        if a >= thea:
            b = 1
        else:
            b = 0
        data_class.append(b)
    return np.asarray(data_class), data_n


def ROC(y, scores, pos_label=1):
    def normalization_min_max(arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
            arr_n[i] = x
        return arr_n

    def get_optimal_threshold(fpr, tpr, thresholds):
        r = list(tpr - fpr)
        return thresholds[r.index(max(r))]

    scores_n = normalization_min_max(scores)
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y, scores_n, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_threshold = get_optimal_threshold(fpr, tpr, thresholds)
    return roc_auc, optimal_threshold


def get_states(input, interval, duration, sample):
    n = int(duration / interval)
    step = int(interval / sample / defaultclock.dt)
    interval_ = int(interval / defaultclock.dt)
    temp = []
    for i in range(n):
        sum = np.sum(input[:, i * interval_: (i + 1) * interval_: step], axis=1)
        temp.append(sum)
    return MinMaxScaler().fit_transform(np.asarray(temp).T)


def load_Data_MNIST(n, path_value, path_label):
    with open(path_value, 'rb') as f1:
        buf1 = f1.read()
    with open(path_label, 'rb') as f2:
        buf2 = f2.read()

    image_index = 0
    image_index += struct.calcsize('>IIII')
    im = []
    for i in range(n):
        temp = struct.unpack_from('>784B', buf1, image_index)
        im.append(np.reshape(temp, (28, 28)))
        image_index += struct.calcsize('>784B')

    label_index = 0
    label_index += struct.calcsize('>II')
    label = np.asarray(struct.unpack_from('>' + str(n) + 'B', buf2, label_index))

    f = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    df = pd.DataFrame({'value': pd.Series(im).apply(f), 'label': pd.Series(label)})
    return df


def get_series_data(n, data_frame, duration, is_order=True, *args, **kwargs):
    try:
        obj = kwargs['obj']
    except KeyError:
        obj = np.arange(10)
    if not is_order:
        data_frame_obj = data_frame[data_frame['label'].isin(obj)].sample(frac=1).reset_index(drop=True)
    else:
        data_frame_obj = data_frame[data_frame['label'].isin(obj)]
    data_frame_s = []
    for value in data_frame_obj['value'][:n]:
        for data in value:
            data_frame_s.append(list(data))
        interval = duration - value.shape[0]
        if interval > 30:
            data_frame_s.extend([[0] * 28] * interval)
        else:
            raise Exception('duration is too short')
    data_frame_s = np.asarray(data_frame_s)
    label = data_frame_obj['label'][:n]
    return data_frame_s, label


# -----parameter setting-------
obj = 1
duration = 100
N_train = 100
N_test = 100
N_pre_train =1000
Dt = defaultclock.dt*2
n = 20
pre_train_loop = 0
sample = 10
I_gain = 5

df_train = load_Data_MNIST(60000, '../../../Data/MNIST_data/train-images.idx3-ubyte',
                               '../../../Data/MNIST_data/train-labels.idx1-ubyte')
df_test = load_Data_MNIST(10000, '../../../Data/MNIST_data/t10k-images.idx3-ubyte',
                               '../../../Data/MNIST_data/t10k-labels.idx1-ubyte')

data_pre_train_s, label_pre_train = get_series_data(N_pre_train, df_train, duration, False, obj=[obj])
data_train_s, label_train = get_series_data(N_train, df_train, duration, False)
data_test_s, label_test = get_series_data(N_test, df_test, duration, False)

duration_pre_train = len(data_pre_train_s) * Dt
duration_train = len(data_train_s) * Dt
duration_test = len(data_test_s) * Dt

equ_in = '''
dv/dt = (I-v) / (1.5*ms) : 1 (unless refractory)
I = stimulus(t,i) : 1
'''

equ = '''
r : 1
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
dg/dt = (-g)/(1.5*ms*r) : 1
dh/dt = (-h)/(1.45*ms*r) : 1
I = tanh(g-h)*I_gain: 1
'''

equ_read = '''
dg/dt = (-g)/(1.5*ms) : 1 
dh/dt = (-h)/(1.45*ms) : 1
I = tanh(g-h)*I_gain : 1
'''

model_STDP = '''
w : 1
wmax : 1
wmin : 1
Apre : 1
Apost = -Apre * taupre / taupost * 1.2 : 1
taupre : second
taupost : second
dapre/dt = -apre/taupre : 1 (clock-driven)
dapost/dt = -apost/taupost : 1 (clock-driven)
'''

on_pre = '''
h+=w
g+=w
'''

on_pre_STDP = '''
h+=w
g+=w
apre += Apre
w = clip(w+apost, wmin, wmax)
'''

on_post_STDP = '''
apost += Apost
w = clip(w+apre, wmin, wmax)
'''

# -----simulation setting-------
Time_array_pre_train = TimedArray(data_pre_train_s, dt=Dt)

Time_array_train = TimedArray(data_train_s, dt=Dt)

Time_array_test = TimedArray(data_test_s, dt=Dt)

Input = NeuronGroup(28, equ_in, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                    name = 'neurongroup_input')

G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                name='neurongroup')

G2 = NeuronGroup(int(n / 4), equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                 name='neurongroup_1')

G_lateral_inh = NeuronGroup(1, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                            name='neurongroup_la_inh')

G_readout = NeuronGroup(n, equ_read, method='euler', name='neurongroup_read')

S = Synapses(Input, G, 'w : 1', on_pre = on_pre ,method='linear', name='synapses')

S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_1')

S3 = Synapses(Input, G_lateral_inh, 'w : 1', on_pre = on_pre ,method='linear', name='synapses_2')

S5 = Synapses(G, G2, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_4')

S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_3')

S6 = Synapses(G_lateral_inh, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_5')

S_readout = Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

# -------network topology----------
S.connect(j='k for k in range(int(n))')
S2.connect(p=0.2)
S3.connect()
S4.connect(p=0.1,condition='i != j')
S5.connect(p=0.2)
S6.connect(j='k for k in range(int(n))')
S_readout.connect(j='i')

S4.wmax = '0.5+rand()*0.4'
S5.wmax = '0.5+rand()*0.4'
S4.wmin = '0.2+rand()*0.2'
S5.wmin = '0.2+rand()*0.2'
S4.Apre = S5.Apre = '0.01'
S4.taupre = S4.taupost ='1*ms+rand()*4*ms'
S5.taupre = S5.taupost ='1*ms+rand()*4*ms'

S.w = 'rand()'
S2.w = '-0.2'
S3.w = '0.8'
S4.w = 'wmin+rand()*(wmax-wmin)'
S5.w = 'wmin+rand()*(wmax-wmin)'
S6.w = '-rand()'

S.delay = '3*ms'
S4.delay = '3*ms'

G.r = '1'
G2.r = '1'
G_lateral_inh.r = '1'

# ------monitor----------------
m_w = StateMonitor(S5, 'w', record=True)
m_w2 = StateMonitor(S4, 'w', record=True)
m_g = StateMonitor(G, (['I', 'v']), record=True)
m_g2 = StateMonitor(G2, (['I', 'v']), record=True)
m_read = StateMonitor(G_readout, ('I'), record=True)
m_inh = StateMonitor(G_lateral_inh, ('I','v'), record=True)
m_in = StateMonitor(Input, ('I','v'), record=True)


# ------create network-------------
net = Network(collect())
net.store('first')
fig_init_w =plt.figure(figsize=(4,4))
brian_plot(S4.w)


###############################################
# ------pre_train------------------
stimulus = Time_array_pre_train
for loop in range(pre_train_loop):
    net.run(duration_pre_train, report='text')

    # ------plot the weight----------------
    fig2 = plt.figure(figsize=(10, 8))
    title('loop: ' + str(loop))
    subplot(211)
    plot(m_w.t / second, m_w.w.T)
    xlabel('Time (s)')
    ylabel('Weight / gmax')
    subplot(212)
    plot(m_w2.t / second, m_w2.w.T)
    xlabel('Time (s)')
    ylabel('Weight / gmax')

    net.store('second')
    net.restore('first')
    S4.w = net._stored_state['second']['synapses_3']['w'][0]
    S5.w = net._stored_state['second']['synapses_4']['w'][0]

# -------change the synapse model----------
S5.pre.code = S4.pre.code = '''
h+=w
g+=w
'''
S5.post.code = S4.post.code = ''


###############################################
# ------run for lms_train-------
del stimulus
stimulus = Time_array_train
net.store('third')
net.run(duration_train, report='text')

# ------lms_train---------------
Y_train = one_versus_the_rest(label_train, selected=np.arange(10))
states = get_states(m_read.I, duration*Dt , duration_train, sample)
P = readout(states, Y)
Y_train_ = lms_test(states,P)
label_train_ = trans_max_to_label(Y_train_)
score_train = accuracy_score(label_train, label_train_)

#####################################
# ----run for test--------
del stimulus
stimulus = Time_array_test
net.restore('third')
net.run(duration_test, report='text')

# -----lms_test-----------
Y_test = one_versus_the_rest(label_test, selected=np.arange(10))
states = get_states(m_read.I, duration*Dt , duration_test, sample)
Y_test_ = lms_test(states,P)
label_test_ = trans_max_to_label(Y_test_)
score_test = accuracy_score(label_test, label_test_)

#####################################
#----------show results-----------
print('Train score: ',score_train)
print('Test score: ',score_test)

#####################################
# ------vis of results----
fig0 = plt.figure(figsize=(20, 6))
subplot(311)
plt.xlim((0, 500))
plot(data_pre_train_s, 'r')
subplot(312)
plt.xlim((0, 500))
plot(data_train_s, 'r')
subplot(313)
plt.xlim((0, 500))
plot(data_test_s, 'r')

fig3 = plt.figure(figsize=(20, 8))
subplot(411)
plt.xlim((0, 100))
plt.plot(m_g.t / ms, m_g.v.T, label='v')
legend(labels=[('V_%s' % k) for k in range(n)], loc='upper right')
subplot(412)
plt.xlim((0, 100))
plt.plot(m_g.t / ms, m_g.I.T, label='I')
legend(labels=[('I_%s' % k) for k in range(n)], loc='upper right')
subplot(413)
plt.xlim((0, 100))
plt.plot(m_in.t / ms, m_in.v.T, label='v')
legend(labels=[('V_%s' % k) for k in range(n)], loc='upper right')
subplot(414)
plt.xlim((0, 100))
plt.plot(m_in.t / ms, m_in.I.T, label='I')
legend(labels=[('I_%s' % k) for k in range(n)], loc='upper right')

fig4 = plt.figure(figsize=(20, 8))
subplot(411)
plt.xlim((0, 100))
plt.plot(m_g2.t / ms, m_g2.v.T, label='v')
legend(labels=[('V_%s' % k) for k in range(n)], loc='upper right')
subplot(412)
plt.xlim((0, 100))
plt.plot(m_g2.t / ms, m_g2.I.T, label='I')
legend(labels=[('I_%s' % k) for k in range(n)], loc='upper right')
subplot(413)
plt.xlim((0, 100))
plt.plot(m_inh.t / ms, m_inh.v.T, label='v')
legend(labels=[('V_%s' % k) for k in range(n)], loc='upper right')
subplot(414)
plt.xlim((0, 100))
plt.plot(m_inh.t / ms, m_inh.I.T, label='I')
legend(labels=[('_%s' % k) for k in range(n)], loc='upper right')


fig5 = plt.figure(figsize=(20, 4))
plt.plot(m_read.t / ms, m_read.I.T, label='I')
legend(labels=[('I_%s' % k) for k in range(n)], loc='upper right')

fig6 =plt.figure(figsize=(4,4))
brian_plot(S4.w)
show()