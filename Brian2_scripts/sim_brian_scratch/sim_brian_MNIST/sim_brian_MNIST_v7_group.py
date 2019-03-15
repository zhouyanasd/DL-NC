# ----------------------------------------
# LSM without STDP for MNIST test
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
import struct
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
data_path = '../../../Data/MNIST_data/'


#-------define brian2 function------------
@check_units(R=1, pre=1, post=1, result=1)
def get_R(R, pre, post):
    return np.array([R[int(pre),int(x)] for x in post])

@check_units(a=1, w=1, result=1)
def gamma(a, w):
    return sp.stats.gamma.rvs(a, size=w.shape)

@check_units(a=1, group=1, result=1)
def gamma_group(a, group):
    return sp.stats.gamma.rvs(a[group.astype(int)])


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
    def __init__(self, duration, dt):
        self.duration = duration
        self.dt = dt
        self.interval = duration * dt

    def get_states(self, input, running_time, sample, normalize=False):
        n = int(running_time / self.interval)
        step = int(self.interval / sample / defaultclock.dt)
        interval_ = int(self.interval / defaultclock.dt)
        temp = []
        for i in range(n):
            sum = np.sum(input[:, i * interval_: (i + 1) * interval_][:, ::-step], axis=1)
            temp.append(sum)
        if normalize:
            return MinMaxScaler().fit_transform(np.asarray(temp)).T
        else:
            return np.asarray(temp).T

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
        V['x'], V['y'], V['z'] = np.meshgrid(np.linspace(0, Y - 1, Y), np.linspace(0, X - 1, X),
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
        if a is None:
            shape = list(b.shape)
            shape[axis] = 0
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b, axis)

    def np_append(self, a, b):
        shape = list(b.shape)
        shape.insert(0, -1)
        if a is None:
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b.reshape(tuple(shape)), axis=0)

    def set_local_parameter_S_group(self, S, parameter, group = 'group', **kwargs):
        if '_post' in parameter:
            S.variables[parameter].set_value(kwargs['value'][S.variables[group].get_value().astype(int)])
        else:
            S.variables[parameter].set_value(kwargs['value'][S.variables[group].get_value().astype(int)][S.j])

    def set_local_group(self, G, group_n, location = False):
        # if location = True, group_n should be (a,b,c)
        # if location = False， group_n should be a
        if not location:
            for g in G:
                g.group = np.sort(list(np.arange(group_n)) * np.ceil(g.N/group_n).astype(int), axis=None)[:g.N]
        else:
            X, Y, Z = [],[],[]
            for g in G:
                X.extend(g.x)
                Y.extend(g.y)
                Z.extend(g.z)
            group = (int(max(X))+1, int(max(Y))+1,int(max(Z))+1)
            g_ = (np.ceil(group[0] / group_n[1]).astype(int), np.ceil(group[1] / group_n[0]).astype(int),
                  np.ceil(group[2] / group_n[2]).astype(int))
            V = np.zeros(group)
            k = np.zeros(group_n, [('x', int), ('y', int), ('z', int)])
            k['x'], k['y'], k['z'] = np.meshgrid(np.arange(0, group[0], g_[0]), np.arange(0, group[1], g_[1]),
                                                 np.arange(0, group[2], g_[2]))
            for index, l in enumerate(k.reshape(np.array(group_n).prod())):
                V[l[0]:l[0] + g_[0], l[1]:l[1] + g_[1], l[2]:l[2] + g_[2]] = int(index)
            for g in G:
                for i in range(g.N):
                    g.group[i] = V[int(g.x[i]),int(g.y[i]), int(g.z[i])]


class Readout():
    def __init__(self, function):
        self.function = function

    def data_init(self, M_train, M_test, label_train, label_test, rate, theta):
        self.rate = rate
        self.theta = theta
        self.iter = 0
        self.X_train = self.add_bis(M_train)
        self.X_test = self.add_bis(M_test)
        self.Y_train = self.prepare_Y(label_train)
        self.Y_test = self.prepare_Y(label_test)
        self.P = np.random.rand(self.X_train.shape[1], self.Y_train.shape[1])
        self.cost_train = 1e+100
        self.cost_test = 1e+100

    def predict_logistic(self, results):
        labels = (results > 0.5).astype(int).T
        return labels

    def calculate_score(self, label, label_predict):
        return [accuracy_score(i, j) for i, j in zip(label, label_predict)]

    def add_bis(self, data):
        one = np.ones((data.shape[1], 1))  # bis
        X = np.hstack((data.T, one))
        return X

    def prepare_Y(self, label):
        if np.asarray(label).ndim == 1:
            return np.asarray([label]).T
        else:
            return np.asarray(label).T

    def cost(self, X, Y, P):
        left = np.multiply(Y, np.log(self.function(X.dot(P))))
        right = np.multiply((1 - Y), np.log(1 - self.function(X.dot(P))))
        return -np.sum(np.nan_to_num(left + right), axis=0) / (len(Y))

    def train(self, X, Y, P):
        P_ = P + X.T.dot(Y - self.function(X.dot(P))) * self.rate
        return P_

    def test(self, X, p):
        return self.function(X.dot(p))

    def stop_condition(self):
        return ((self.cost_train - self.cost(self.X_train, self.Y_train, self.P)) > self.theta).any() and \
               ((self.cost_test - self.cost(self.X_test, self.Y_test, self.P)) > self.theta).any() or self.iter < 100

    def readout(self):
        self.iter = 0
        while self.stop_condition():
            self.iter += 1
            self.cost_train = self.cost(self.X_train, self.Y_train, self.P)
            self.cost_test = self.cost(self.X_test, self.Y_test, self.P)
            self.P = self.train(self.X_train, self.Y_train, self.P)
            if self.iter % 10000 == 0:
                print(self.iter, self.cost_train, self.cost_test)
        print(self.iter, self.cost_train, self.cost_test)
        return self.test(self.X_train, self.P), self.test(self.X_test, self.P)

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


class MNIST_classification(Base):
    def __init__(self, shape, duration, dt):
        super().__init__(duration, dt)
        self.shape = shape

    def load_Data_MNIST(self, n, path_value, path_label, is_norm=True):
        with open(path_value, 'rb') as f1:
            buf1 = f1.read()
        with open(path_label, 'rb') as f2:
            buf2 = f2.read()

        image_index = 0
        image_index += struct.calcsize('>IIII')
        im = []
        for i in range(n):
            temp = struct.unpack_from('>784B', buf1, image_index)
            im.append(np.reshape(temp, self.shape))
            image_index += struct.calcsize('>784B')

        label_index = 0
        label_index += struct.calcsize('>II')
        label = np.asarray(struct.unpack_from('>' + str(n) + 'B', buf2, label_index))
        if is_norm:
            f = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
            df = pd.DataFrame({'value': pd.Series(im).apply(f), 'label': pd.Series(label)})
        else:
            df = pd.DataFrame({'value': pd.Series(im), 'label': pd.Series(label)})
        return df

    def load_Data_MNIST_all(self, path, is_norm=True):
        self.train = self.load_Data_MNIST(60000, path + 'train-images.idx3-ubyte',
                                          path + 'train-labels.idx1-ubyte', is_norm)
        self.test = self.load_Data_MNIST(10000, path + 't10k-images.idx3-ubyte',
                                         path + 't10k-labels.idx1-ubyte', is_norm)

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

    def _encoding_cos_rank(self, x, n, A):
        encoding = np.zeros((x.shape[0] * A, n * x.shape[1]), dtype='<i1')
        for i in range(int(n)):
            trans_cos = np.around(0.5 * A * (np.cos(x + np.pi * (i / n)) + 1)).clip(0, A - 1)
            for index_0, p in enumerate(trans_cos):
                for index_1, q in enumerate(p):
                    encoding[int(q)+ A * index_0, index_1 * n + i] = 1
        return encoding

    def _encoding_cos_rank_ignore_0(self, x, n, A):
        encoding = np.zeros((x.shape[0] * A, n * x.shape[1]), dtype='<i1')
        for i in range(int(n)):
            trans_cos = np.around(0.5 * A * (np.cos(x + np.pi * (i / n)) + 1)).clip(0, A - 1)
            encoded_zero = int(np.around(0.5 * A * (np.cos(0 + np.pi * (i / n)) + 1)).clip(0, A - 1))
            for index_0, p in enumerate(trans_cos):
                for index_1, q in enumerate(p):
                    if int(q) == encoded_zero:
                        continue
                    else:
                        encoding[int(q)+ A * index_0, index_1 * n + i] = 1
        return encoding

    def encoding_latency_MNIST(self, coding_f, analog_data, coding_n, min=0, max=np.pi):
        f = lambda x: (max - min) * (x - np.min(x)) / (np.max(x) - np.min(x))
        coding_duration = self.duration / self.shape[0]
        if (coding_duration - int(coding_duration)) == 0.0:
            value = analog_data['value'].apply(f).apply(coding_f, n=coding_n, A=int(coding_duration))
            return pd.DataFrame({'value': pd.Series(value), 'label': pd.Series(analog_data['label'])})
        else:
            raise ValueError('duration must divide (coding_n*length of data) exactly')

    def get_series_data(self, data_frame, is_group=False):
        data_frame_s = None
        if not is_group:
            for value in data_frame['value']:
                data_frame_s = self.np_extend(data_frame_s, value, 0)
        else:
            for value in data_frame['value']:
                data_frame_s = self.np_append(data_frame_s, value)
        label = data_frame['label']
        return data_frame_s, label

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


#--------define network run function-------------------
def run_net(inputs):
    states = None
    monitor_record= {
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
        net.run(duration * Dt)
        states = base.np_append(states, G_readout.variables['v'].get_value())
        if Switch_monitor :
            monitor_record= base.update_states('numpy', m_g_ex.I, m_g_ex.v, m_g_in.I, m_g_in.v, m_read.I,
                                                m_read.v, m_input.I, **monitor_record)
        net.restore('init')
    return (MinMaxScaler().fit_transform(states)).T, monitor_record


###################################
#--------switch setting--------
Switch_monitor = True

# -----parameter setting-------
coding_n = 10
MNIST_shape = (28, 28)
coding_duration = 10
duration = coding_duration*MNIST_shape[0]
F_train = 0.05
F_test = 0.05
Dt = defaultclock.dt = 1*ms

group = (2, 2, 4)
group_ = np.prod(np.array(group))

n_ex = 400
n_inh = int(n_ex/4)
n_input = MNIST_shape[1]*coding_n
n_read = n_ex+n_inh

R = 2

f_in = rand(group_)
f_EE = rand(group_)
f_EI = rand(group_)
f_IE = rand(group_)
f_II = rand(group_)

A_EE = 30*f_EE
A_EI = 60*f_EI
A_IE = 19*f_IE
A_II = 19*f_II
A_inE = 18*f_in
A_inI = 9*f_in

tau_ex = np.arange(group_)*10
tau_inh = np.arange(group_)*10
tau_read = 30

p_inE = 0.1
p_inI = 0.1

###########################################
#-------class initialization----------------------
function = Function()
base = Base(duration, Dt)
readout = Readout(function.logistic)
result = Result()
MNIST = MNIST_classification(MNIST_shape, duration, Dt)

#-------data initialization----------------------
MNIST.load_Data_MNIST_all(data_path)
df_train = MNIST.select_data(F_train, MNIST.train)
df_test = MNIST.select_data(F_test, MNIST.test)

df_en_train = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_train, coding_n)
df_en_test = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_test, coding_n)

data_train_s, label_train = MNIST.get_series_data_list(df_en_train, is_group = True)
data_test_s, label_test = MNIST.get_series_data_list(df_en_test, is_group = True)

#------definition of equation-------------
neuron_in = '''
I = stimulus(t,i) : 1
'''

neuron = '''
tau : 1
dv/dt = (I-v) / (30*ms) : 1 (unless refractory)
dg/dt = (-g)/(3*ms) : 1
dh/dt = (-h)/(6*ms) : 1
I = (g+h)+13.5: 1
x : 1
y : 1
z : 1
group : 1 
'''

neuron_read = '''
tau : 1
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
                    name = 'neurongroup_input')

G_ex = NeuronGroup(n_ex, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                name ='neurongroup_ex')

G_inh = NeuronGroup(n_inh, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                name ='neurongroup_in')

G_readout = NeuronGroup(n_read, neuron_read, method='euler', name='neurongroup_read')

S_inE = Synapses(Input, G_ex, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_inE')

S_inI = Synapses(Input, G_inh, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_inI')

S_EE = Synapses(G_ex, G_ex, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_EE')

S_EI = Synapses(G_ex, G_inh, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_EI')

S_IE = Synapses(G_inh, G_ex, synapse, on_pre = on_pre_inh ,method='euler', name='synapses_IE')

S_II = Synapses(G_inh, G_inh, synapse, on_pre = on_pre_inh ,method='euler', name='synapses_I')

S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_ex, method='euler')

S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_inh, method='euler')

#-------location and group----------
[G_ex,G_in] = base.allocate([G_ex,G_inh],5,5,20)
base.set_local_group((G_ex,G_inh),group,True)

#-------initialization of neuron parameters----------
G_ex.v = '13.5+1.5*rand()'
G_inh.v = '13.5+1.5*rand()'
G_readout.v = '0'
G_ex.g = '0'
G_inh.g = '0'
G_readout.g = '0'
G_ex.h = '0'
G_inh.h = '0'
G_readout.h = '0'
base.set_local_parameter_S_group(S_EE, 'tau_post', 'group', value = tau_ex)
base.set_local_parameter_S_group(S_II, 'tau_post', 'group', value = tau_inh)
G_readout.tau = tau_read

# -------initialization of network topology and synapses parameters----------
S_EE.variables.add_dynamic_array('R_', size=(group_,group_))
S_EI.variables.add_dynamic_array('R_', size=(group_,group_))
S_IE.variables.add_dynamic_array('R_', size=(group_,group_))
S_II.variables.add_dynamic_array('R_', size=(group_,group_))
S_EE.variables['R_'].set_value(np.diag(rand(group_)*R))
S_EI.variables['R_'].set_value(np.diag(rand(group_)*R))
S_IE.variables['R_'].set_value(np.diag(rand(group_)*R))
S_II.variables['R_'].set_value(np.diag(rand(group_)*R))

S_inE.connect(condition='j<0.3*N_post', p = p_inE)
S_inI.connect(condition='j<0.3*N_post', p = p_inI)
S_EE.connect(condition='(i != j)',
             p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/get_R(R_,group_pre,group_post)**2)')
S_EI.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/get_R(R_,group_pre,group_post)**2)')
S_IE.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/get_R(R_,group_pre,group_post)**2)')
S_II.connect(condition='i != j',
             p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/get_R(R_,group_pre,group_post)**2)')
S_E_readout.connect(j='i')
S_I_readout.connect(j='i+n_ex')

S_inE.variables.add_array('A', size=group_, values=A_inE)
S_inI.variables.add_array('A', size=group_, values=A_inI)
S_EE.variables.add_array('A', size=group_, values=A_EE)
S_IE.variables.add_array('A', size=group_, values=A_IE)
S_EI.variables.add_array('A', size=group_, values=A_EI)
S_II.variables.add_array('A', size=group_, values=A_II)

S_inE.w = 'gamma_group(A, group_post)'
S_inI.w = 'gamma_group(A, group_post)'
S_EE.w = 'gamma_group(A, group_post)'
S_IE.w = 'gamma_group(A, group_post)'
S_EI.w = 'gamma_group(A, group_post)'
S_II.w = 'gamma_group(A, group_post)'

S_EE.pre.delay = '1.5*ms'
S_EI.pre.delay = '0.8*ms'
S_IE.pre.delay = '0.8*ms'
S_II.pre.delay = '0.8*ms'

# --------monitors setting----------
if Switch_monitor :
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
#----------show results-----------
print('Train score: ',score_train)
print('Test score: ',score_test)


#####################################
#-----------save monitor data-------
if Switch_monitor :
    result.result_save('monitor_train.pkl', **monitor_record_train)
    result.result_save('monitor_test.pkl', **monitor_record_test)
result.result_save('states_records.pkl', states_train = states_train, states_test = states_test)

#####################################
# ------vis of results----
fig_init_w =plt.figure(figsize=(16,16))
subplot(421)
brian_plot(S_EE.w)
subplot(422)
brian_plot(S_EI.w)
subplot(423)
brian_plot(S_IE.w)
subplot(424)
brian_plot(S_II.w)
show()

#-------for animation in Jupyter-----------
monitor = result.result_pick('monitor_test.pkl')
play, slider, fig = result.animation(np.arange(monitor['m_read.v'].shape[1]), monitor['m_read.v'], duration, 10*duration)
widgets.VBox([widgets.HBox([play, slider]),fig])