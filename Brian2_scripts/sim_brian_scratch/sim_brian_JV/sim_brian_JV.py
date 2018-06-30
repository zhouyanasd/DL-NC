# ----------------------------------------
# LSM without STDP for Jv test
# add neurons to readout layer for multi-classification(one-versus-the-rest)
# using softmax(logistic regression)
# input layer is changed to 781*1 with encoding method
# change the LSM structure according to Maass paper
# new calculate flow as Maass_ST
# simplify the coding method with only extend the rank
# ----------------------------------------

from brian2 import *
from brian2tools import *
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pickle
from bqplot import *
import ipywidgets as widgets
import warnings
import os

warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)
data_path = '../../../Data/Jv/'


# ------define general function------------
class Function():
    def __init__(self):
        pass

    def logistic(self, f):
        return 1 / (1 + np.exp(-f))

    def softmax(self, z):
        return np.array([(np.exp(i) / np.sum(np.exp(i))) for i in z])

    def gamma(self, a, size):
        return sp.stats.gamma.rvs(a, size=size)


class Base():
    def update_states(self, type='pandas', *args, **kwargs):
        for seq, state in enumerate(kwargs):
            if type == 'pandas':
                kwargs[state] = kwargs[state].append(pd.DataFrame(args[seq]))
            elif type == 'numpy':
                kwargs[state] = self.np_extend(kwargs[state], args[seq], 1)
        return kwargs

    def normalization_min_max(self, arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
            arr_n[i] = x
        return arr_n

    def mse(self, y_test, y):
        return sp.sqrt(sp.mean((y_test - y) ** 2))

    def classification(self, thea, data):
        data_n = self.normalization_min_max(data)
        data_class = []
        for a in data_n:
            if a >= thea:
                b = 1
            else:
                b = 0
            data_class.append(b)
        return np.asarray(data_class), data_n

    def allocate(self, G, X, Y, Z):
        V = np.zeros((X, Y, Z), [('x', float), ('y', float), ('z', float)])
        V['x'], V['y'], V['z'] = np.meshgrid(np.linspace(0, X - 1, X), np.linspace(0, X - 1, X),
                                             np.linspace(0, Z - 1, Z))
        V = V.reshape(X * Y * Z)
        np.random.shuffle(V)
        n = 0
        for g in G:
            for i in range(g.N):
                g.x[i], g.y[i], g.z[i] = V[n][0], V[n][1], V[n][2]
                n += 1
        return G

    def w_norm2(self, n_post, Synapsis):
        for i in range(n_post):
            a = Synapsis.w[np.where(Synapsis._synaptic_post == i)[0]]
            Synapsis.w[np.where(Synapsis._synaptic_post == i)[0]] = a / np.linalg.norm(a)

    def np_extend(self, a, b, axis=0):
        if a == None:
            shape = list(b.shape)
            shape[axis] = 0
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b, axis)

    def np_append(self, a, b):
        shape = list(b.shape)
        shape.insert(0, -1)
        if a == None:
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b.reshape(tuple(shape)), axis=0)


class Readout():
    def readout_sk(self, X_train, X_test, y_train, y_test, **kwargs):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(**kwargs)
        lr.fit(X_train.T, y_train.T)
        y_train_predictions = lr.predict(X_train.T)
        y_test_predictions = lr.predict(X_test.T)
        return accuracy_score(y_train_predictions, y_train.T), accuracy_score(y_test_predictions, y_test.T)


class Result():
    def __init__(self):
        pass

    def result_save(self, path, *arg, **kwarg):
        if os.path.exists(path):
            os.remove(path)
        fw = open(path, 'wb')
        pickle.dump(kwarg, fw)
        fw.close()

    def result_pick(self, path):
        fr = open(path, 'rb')
        data = pickle.load(fr)
        fr.close()
        return data

    def animation(self, t, v, interval, duration, a_step=10, a_interval=100, a_duration=10):
        xs = LinearScale()
        ys = LinearScale()
        line = Lines(x=t[:interval], y=v[:, :interval], scales={'x': xs, 'y': ys})
        xax = Axis(scale=xs, label='x', grid_lines='solid')
        yax = Axis(scale=ys, orientation='vertical', tick_format='0.2f', label='y', grid_lines='solid')
        fig = Figure(marks=[line], axes=[xax, yax], animation_duration=a_duration)

        def on_value_change(change):
            line.x = t[change['new']:interval + change['new']]
            line.y = v[:, change['new']:interval + change['new']]

        play = widgets.Play(
            interval=a_interval,
            value=0,
            min=0,
            max=duration,
            step=a_step,
            description="Press play",
            disabled=False
        )
        slider = widgets.IntSlider(min=0, max=duration)
        widgets.jslink((play, 'value'), (slider, 'value'))
        slider.observe(on_value_change, names='value')
        return play, slider, fig


class Jv_classification():
    def __init__(self, coding_duration):
        self.coding_duration = coding_duration

    def load_Data_Jv(self, t, path_value, path_label, is_norm=True):
        if t == "train":
            label = np.loadtxt(path_label, delimiter=None).astype(int)[1]
        elif t == "test":
            label = np.loadtxt(path_label, delimiter=None).astype(int)[0]
        else:
            raise TypeError("t must be 'train' or 'test'")
        data = np.loadtxt(path_value, delimiter=None)
        if is_norm:
            data = MinMaxScaler().fit_transform(data)
        s = open(path_value, 'r')
        i = -1
        size = []
        while True:
            lines = s.readline()
            i += 1
            if not lines:
                break
            if lines == '\n':  # "\n" needed to be added at the end of the file
                i -= 1
                size.append(i)
                continue
        size_d = np.concatenate(([0], (np.asarray(size) + 1)))
        data_list = [data[size_d[i]:size_d[i + 1]] for i in range(len(size_d) - 1)]
        label_list = []
        j = 0
        for n in label:
            label_list.extend([j] * n)
            j += 1
        data_frame = pd.DataFrame({'value': pd.Series(data_list), 'label': pd.Series(label_list)})
        return data_frame

    def load_Data_Jv_all(self, path, is_norm=True):
        self.train = self.load_Data_Jv('train', path + 'train.txt',
                                       path + 'size.txt', is_norm)
        self.test = self.load_Data_Jv('test', path + 'test.txt',
                                      path + 'size.txt', is_norm)

    def select_data(self, fraction, data_frame, is_order=True, **kwargs):
        try:
            selected = kwargs['selected']
        except KeyError:
            selected = np.arange(10)
        if is_order:
            data_frame_selected = data_frame[data_frame['label'].isin(selected)].sample(
                frac=fraction).sort_index().reset_index(drop=True)
        else:
            data_frame_selected = data_frame[data_frame['label'].isin(selected)].sample(frac=fraction).reset_index(
                drop=True)
        return data_frame_selected

    def _encoding_cos(self, x, n, A):
        encoding = []
        for i in range(int(n)):
            trans_cos = np.around(0.5 * A * (np.cos(x + np.pi * (i / n)) + 1)).clip(0, A - 1)
            coding = [([0] * trans_cos.shape[1]) for i in range(A * trans_cos.shape[0])]
            for index_0, p in enumerate(trans_cos):
                for index_1, q in enumerate(p):
                    coding[int(q) + A * index_0][index_1] = 1
            encoding.extend(coding)
        return np.asarray(encoding)

    def _encoding_cos_rank(self, x, n, A):
        encoding = np.zeros((x.shape[0] * A, n * x.shape[1]), dtype='<i1')
        for i in range(int(n)):
            trans_cos = np.around(0.5 * A * (np.cos(x + np.pi * (i / n)) + 1)).clip(0, A - 1)
            for index_0, p in enumerate(trans_cos):
                for index_1, q in enumerate(p):
                    encoding[int(q) + A * index_0, index_1 * n + i] = 1
        return np.asarray(encoding)

    def encoding_latency_Jv(self, coding_f, analog_data, coding_n, min=0, max=np.pi):
        f = lambda x: (max - min) * (x - np.min(x)) / (np.max(x) - np.min(x))
        value = analog_data['value'].apply(f).apply(coding_f, n=coding_n, A=int(self.coding_duration))
        return pd.DataFrame({'value': pd.Series(value), 'label': pd.Series(analog_data['label'])})

    def get_series_data_list(self, data_frame, is_group=False):
        data_frame_s = []
        if not is_group:
            for value in data_frame['value']:
                data_frame_s.extend(value)
        else:
            for value in data_frame['value']:
                data_frame_s.append(value)
        label = data_frame['label']
        return np.asarray(data_frame_s), label


# --------define network run function-------------------
Switch_monitor = True


def run_net(inputs):
    states = None
    monitor_record = {
        'm_g_ex.I': None,
        'm_g_ex.v': None,
        'm_g_in.I': None,
        'm_g_in.v': None,
        'm_read.I': None,
        'm_read.v': None,
        'm_input.I': None}
    for ser, data in enumerate(inputs):
        if ser % 50 == 0:
            print('The simulation is running at %s time.' % ser)
        stimulus = TimedArray(data, dt=Dt)
        duration = data.shape[0]
        net.run(duration * Dt)
        states = base.np_append(states, G_readout.variables['v'].get_value())
        if record:
            monitor_record = base.update_states('numpy', m_g_ex.I, m_g_ex.v, m_g_in.I, m_g_in.v, m_read.I,
                                                m_read.v, m_input.I, **monitor_record)
        net.restore('init')
    return (MinMaxScaler().fit_transform(states)).T, monitor_record


###################################
# -----parameter setting-------
coding_n = 10
dim = 12
coding_duration = 10
F_train = 1
F_test = 1
Dt = defaultclock.dt = 1 * ms

pre_train_loop = 0

n_ex = 400
n_inh = int(n_ex / 4)
n_input = dim * coding_n
n_read = n_ex + n_inh

R = 2

A_EE = 30
A_EI = 60
A_IE = 19
A_II = 19
A_inE = 18
A_inI = 9

p_inE = 0.1
p_inI = 0.1

###########################################
# -------class initialization----------------------
function = Function()
base = Base()
readout = Readout(function.logistic)
result = Result()
Jv = Jv_classification(coding_duration)

# -------data initialization----------------------
Jv.load_Data_Jv_all(data_path)
df_train = Jv.select_data(F_train, Jv.train, False)
df_test = Jv.select_data(F_test, Jv.test, False)

df_en_train = Jv.encoding_latency_Jv(Jv._encoding_cos_rank, df_train, coding_n)
df_en_test = Jv.encoding_latency_Jv(Jv._encoding_cos_rank, df_test, coding_n)

data_train_s, label_train = Jv.get_series_data_list(df_en_train, is_group=True)
data_test_s, label_test = Jv.get_series_data_list(df_en_test, is_group=True)

# ------definition of equation-------------
neuron_in = '''
I = stimulus(t,i) : 1
'''

neuron = '''
dv/dt = (I-v) / (30*ms) : 1 (unless refractory)
dg/dt = (-g)/(3*ms) : 1
dh/dt = (-h)/(6*ms) : 1
I = (g+h)+13.5: 1
x : 1
y : 1
z : 1
'''

neuron_read = '''
dv/dt = (I-v) / (30*ms) : 1
dg/dt = (-g)/(3*ms) : 1 
dh/dt = (-h)/(6*ms) : 1
I = (g+h): 1
'''

synapse = '''
w : 1
'''

on_pre_ex = '''
g+=w
'''

on_pre_inh = '''
h-=w
'''

# -----Neurons and Synapses setting-------
Input = NeuronGroup(n_input, neuron_in, threshold='I > 0', method='euler', refractory=0 * ms,
                    name='neurongroup_input')

G_ex = NeuronGroup(n_ex, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2.99 * ms,
                   name='neurongroup_ex')

G_inh = NeuronGroup(n_inh, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=1.99 * ms,
                    name='neurongroup_in')

G_readout = NeuronGroup(n_read, neuron_read, method='euler', name='neurongroup_read')

S_inE = Synapses(Input, G_ex, synapse, on_pre=on_pre_ex, method='euler', name='synapses_inE')

S_inI = Synapses(Input, G_inh, synapse, on_pre=on_pre_ex, method='euler', name='synapses_inI')

S_EE = Synapses(G_ex, G_ex, synapse, on_pre=on_pre_ex, method='euler', name='synapses_EE')

S_EI = Synapses(G_ex, G_inh, synapse, on_pre=on_pre_ex, method='euler', name='synapses_EI')

S_IE = Synapses(G_inh, G_ex, synapse, on_pre=on_pre_inh, method='euler', name='synapses_IE')

S_II = Synapses(G_inh, G_inh, synapse, on_pre=on_pre_inh, method='euler', name='synapses_I')

S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_ex, method='euler')

S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_inh, method='euler')

# -------initialization of neuron parameters----------
G_ex.v = '13.5+1.5*rand()'
G_inh.v = '13.5+1.5*rand()'
G_readout.v = '0'
G_ex.g = '0'
G_inh.g = '0'
G_readout.g = '0'
G_ex.h = '0'
G_inh.h = '0'
G_readout.h = '0'

[G_ex, G_in] = base.allocate([G_ex, G_inh], 5, 5, 20)

# -------initialization of network topology and synapses parameters----------
S_inE.connect(condition='j<0.3*N_post', p=p_inE)
S_inI.connect(condition='j<0.3*N_post', p=p_inI)
S_EE.connect(condition='i != j', p='0.3*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_EI.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_IE.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_II.connect(condition='i != j', p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_E_readout.connect(j='i')
S_I_readout.connect(j='i+n_ex')

S_inE.w = function.gamma(A_inE, S_inE.w.shape)
S_inI.w = function.gamma(A_inI, S_inI.w.shape)
S_EE.w = function.gamma(A_EE, S_EE.w.shape)
S_IE.w = function.gamma(A_IE, S_IE.w.shape)
S_EI.w = function.gamma(A_EI, S_EI.w.shape)
S_II.w = function.gamma(A_II, S_II.w.shape)

S_EE.delay = '1.5*ms'
S_EI.delay = '0.8*ms'
S_IE.delay = '0.8*ms'
S_II.delay = '0.8*ms'

# --------monitors setting----------
if Switch_monitor:
    m_g_in = StateMonitor(G_in, (['I', 'v']), record=True)
    m_g_ex = StateMonitor(G_ex, (['I', 'v']), record=True)
    m_read = StateMonitor(G_readout, (['I', 'v']), record=True)
    m_input = StateMonitor(Input, ('I'), record=True)

# ------create network-------------
net = Network(collect())
net.store('init')

###############################################
# ------run for train-------
states_train, monitor_record_train = run_net(data_train_s)

# ----run for test--------
states_test, monitor_record_test = run_net(data_test_s)

#####################################
# ------Readout---------------
score_train, score_test = readout.readout_sk(states_train, states_test, label_train, label_test, solver="lbfgs",
                                             multi_class="multinomial")

#####################################
# ----------show results-----------
print('Train score: ', score_train)
print('Test score: ', score_test)

#####################################
# -----------save monitor data-------
if Switch_monitor:
    result.result_save('monitor_train.pkl', **monitor_record_train)
    result.result_save('monitor_test.pkl', **monitor_record_test)
result.result_save('states_records.pkl', states_train=states_train, states_test=states_test)

#####################################
# ------vis of results----
fig_init_w = plt.figure(figsize=(16, 16))
subplot(421)
brian_plot(S_EE.w)
subplot(422)
brian_plot(S_EI.w)
subplot(423)
brian_plot(S_IE.w)
subplot(424)
brian_plot(S_II.w)
show()

# -------for animation in Jupyter-----------
monitor = result.result_pick('monitor_test.pkl')
play, slider, fig = result.animation(np.arange(monitor['m_read.v'].shape[1]), monitor['m_read.v'], coding_duration * 20,
                                     20 * 20 * coding_duration)
