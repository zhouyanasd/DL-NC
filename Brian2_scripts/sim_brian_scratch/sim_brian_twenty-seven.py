# ----------------------------------------
# LSM with STDP for JV test
# add Input layer as input and the encoding is transformed into spike trains
# simulation 7--analysis 3
# ----------------------------------------

from brian2 import *
from brian2tools import *
from scipy.optimize import leastsq
import scipy as sp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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


def label_to_obj(label, obj):
    temp = []
    for a in label:
        if a == obj:
            temp.append(1)
        else:
            temp.append(0)
    return np.asarray(temp)


def load_Data_JV(t, path_value="../../Data/jv/train.txt", path_label="../../Data/jv/size.txt"):
    if t == "train":
        label = np.loadtxt(path_label, delimiter=None).astype(int)[1]
    elif t == "test":
        label = np.loadtxt(path_label, delimiter=None).astype(int)[0]
    else:
        raise TypeError("t must be 'train' or 'test'")
    data = np.loadtxt(path_value, delimiter=None)
    data = MinMaxScaler().fit_transform(data)
    s = open(path_value, 'r')
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
    size_d = np.asarray(size_d) + 1
    size_d = np.concatenate(([0], size_d))
    data_list = [data[size_d[i]:size_d[i + 1]] for i in range(len(size_d) - 1)]
    label_list = []
    j = 0
    for n in label:
        label_list.extend([j] * n)
        j += 1
    data_frame = pd.DataFrame({'value': pd.Series(data_list), 'label': pd.Series(label_list)})
    return data_frame


def get_series_data(data_frame, duration, is_order = True, *args, **kwargs):
    try:
        obj = kwargs['obj']
    except KeyError:
        obj = np.arange(9)
    if not is_order:
        data_frame_obj = data_frame[data_frame['label'].isin(obj)].sample(frac=1).reset_index(drop=True)
    else:
        data_frame_obj = data_frame[data_frame['label'].isin(obj)]
    data_frame_s = []
    for value in data_frame_obj['value']:
        for data in value:
            data_frame_s.append(list(data))
        interval = duration - value.shape[0]
        if interval >30 :
            data_frame_s.extend([[0]*12]*interval)
        else:
            raise Exception('duration is too short')
    data_frame_s = np.asarray(data_frame_s)
    label = data_frame_obj['label']
    return data_frame_s, label


# -----parameter setting-------
data_train = load_Data_JV(t='train',path_value="../../Data/jv/train.txt")
data_test = load_Data_JV(t='test',path_value="../../Data/jv/test.txt")

obj = 1
duration = 100
dt = defaultclock.dt*2
data_train_s , label_train = get_series_data(data_train,duration)
data_test_s , label_test = get_series_data(data_test,duration)
data_pre_s , label_pre = get_series_data(data_train,duration, False, obj=[obj])

duration_train = len(data_train_s) * dt
duration_test = len(data_test_s) * dt
duration_pre = len(data_pre_s) * dt

n = 20
pre_train_loop = 0
threshold = 0.5

equ_in = '''
dv/dt = (I-v) / (1.5*ms) : 1 (unless refractory)
I = stimulus(t,i) : 1
'''

equ = '''
r : 1
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
dg/dt = (-g)/(1.5*ms*r) : 1
dh/dt = (-h)/(1.45*ms*r) : 1
I = tanh(g-h)*20: 1
'''

equ_read = '''
dg/dt = (-g)/(1.5*ms) : 1 
dh/dt = (-h)/(1.45*ms) : 1
I = tanh(g-h)*20 : 1
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
Time_array_train = TimedArray(data_train_s, dt=dt)

Time_array_test = TimedArray(data_test_s, dt=dt)

Time_array_pre = TimedArray(data_pre_s, dt=dt)

Input = NeuronGroup(12, equ_in, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
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
S.connect(j='k for k in range(int(n*0.1))')
S2.connect(p=0.2)
S3.connect()
S4.connect(p=0.1,condition='i != j')
S5.connect(p=0.2)
S6.connect(j='k for k in range(int(n*0.1))')
S_readout.connect(j='i')

S4.wmax = '0.5+rand()*0.5'
S5.wmax = '0.5+rand()*0.5'
S4.wmin = 'rand()*0.5'
S5.wmin = 'rand()*0.5'
S4.Apre = S5.Apre = '0.01'
S4.taupre = S4.taupost ='1*ms+rand()*4*ms'
S5.taupre = S5.taupost ='1*ms+rand()*4*ms'

S.w = '1.4+j*'+str(0.6/(n*0.1))
S2.w = '-0.2'
S3.w = '0.8'
S4.w = 'wmin+rand()*(wmax-wmin)'
S5.w = 'wmin+rand()*(wmax-wmin)'
S6.w = [-0.01, -1.4]

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
stimulus = Time_array_pre
for loop in range(pre_train_loop):
    net.run(duration_pre)

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
del stimulus
stimulus = Time_array_train

S5.pre.code = S4.pre.code = '''
h+=w
g+=w
'''
S5.post.code = S4.post.code = ''


###############################################
# ------run for lms_train-------
net.store('third')
net.run(duration)