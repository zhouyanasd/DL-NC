# ----------------------------------------
# LSM with STDP for MNIST test with in-E plasticity by category
# add neurons to readout layer for multi-classification(one-versus-the-rest)
# using softmax(logistic regression)
# input layer is changed to 781*1 with encoding method
# change the LSM structure according to Maass paper
# new calculate flow as Maass_ST
# simplify the coding method with only extend the rank
# for the grad search
# ----------------------------------------

from brian2 import *
import scipy as sp
import struct
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pickle
from bqplot import *
import ipywidgets as widgets
import warnings
import os
from multiprocessing import Pool
import logging


logging.file_log = False
warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)
data_path = '../../../Data/MNIST_data/'


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

    def update_metrics(self, type='numpy', *args, **kwargs):
        for seq, state in enumerate(kwargs):
            if type == 'list':
                kwargs[state].append(args[seq])
            elif type == 'numpy':
                kwargs[state] = self.np_append(kwargs[state], args[seq])
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

    def connection_matrix(self, n_pre, n_post, sources, targets, values):
        full_matrix = np.zeros((n_pre, n_post))
        full_matrix[targets, sources] = values
        return full_matrix

    def spectral_radius(self, S, is_norm = False):
        if isinstance(S, Synapses):
            n_pre = S.N_pre
            n_post = S.N_post
            sources = S.i[:]
            targets = S.j[:]
            values = S.w[:] - np.mean(S.variables['w'].get_value())
            if n_pre== n_post:
                ma = self.connection_matrix(n_pre, n_post, sources, targets, values)
                if is_norm :
                    ma = ma /np.sqrt(np.var(ma))/np.sqrt(n_post)
                else:
                    ma = ma /np.sqrt(n_post)
            else:
                return np.array(-1)
            a, b = np.linalg.eig(ma)
            return np.max(np.abs(a))
        else:
            raise ('The input need to be a object of Synapses')

    def get_plasticity_confuse(self, metric_plasticity_list, label):
        dis = []
        for metric_plasticity in metric_plasticity_list:
            df = pd.DataFrame({'weight_changed': list(metric_plasticity['weight_changed']),
                               'label': label})
            df_a = df[:int(0.5 * df.index.size)]
            df_b = df[int(0.5 * df.index.size):]
            df_a_divide = [df_a[df_a['label'] == x] for x in
                           df_a['label'].value_counts().index.sort_values().tolist()]
            df_b_divide = [df_b[df_b['label'] == x] for x in
                           df_b['label'].value_counts().index.sort_values().tolist()]
            n = len(df['label'].value_counts().index.sort_values().tolist())
            _dis = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    _dis[i][j] = numpy.linalg.norm(
                        np.array(list(df_a_divide[i]['weight_changed'])).mean(axis=0) -
                        np.array(list(df_b_divide[j]['weight_changed'])).mean(axis=0))
            dis.append(_dis)
        return dis

    def get_confusion(self, confuse_matrix):
        return [np.abs((matrix - np.diag(np.diag(matrix))).mean() - np.diag(matrix).mean())/matrix.mean()
                for matrix in confuse_matrix]

    def set_local_parameter(self, S, parameter, boundary, method='random', **kwargs):
        if method == 'random':
            random = rand(S.N_post) * (boundary[1]-boundary[0]) + boundary[0]
            if '_post' in parameter:
                S.variables[parameter].set_value(random)
            else:
                S.variables[parameter].set_value(random[S.j])
        if method == 'group':
            group_n = kwargs['group_n']
            n = int(np.floor(S.N_post / group_n))
            random = zeros(S.N_post)
            for i in range(group_n):
                try:
                    random[i * n:(i + 1) * n] = rand() * (boundary[1]-boundary[0]) + boundary[0]
                except IndexError:
                    random[i * n:] = rand() * (boundary[1]-boundary[0]) + boundary[0]
                    continue
            if '_post' in parameter:
                S.variables[parameter].set_value(random)
            else:
                S.variables[parameter].set_value(random[S.j])
        if method == 'location':
            group_n = kwargs['group_n']
            location_label = kwargs['location_label']
            random = zeros(S.N_post)
            bound = np.linspace(0, max(S.variables[location_label].get_value() + 1), num=group_n + 1)
            for i in range(group_n):
                random[(S.variables[location_label].get_value() >= bound[i]) & (
                            S.variables[location_label].get_value() < bound[i + 1])] \
                    = rand() * (boundary[1]-boundary[0]) + boundary[0]
            if '_post' in parameter:
                S.variables[parameter].set_value(random)
            else:
                S.variables[parameter].set_value(random[S.j])
        if method == 'in_coming':
            max_incoming = max(S.N_incoming)
            random = S.N_incoming / max_incoming * (boundary[1]-boundary[0]) + boundary[0]
            if '_post' in parameter:
                S.variables[parameter].set_value(random)
            else:
                S.variables[parameter].set_value(random[S.j])

    def parameters_GS(self, *args, **kwargs):
        #---------------
        # args = [(min,max),]
        # kwargs = {'parameter' = number，}
        #---------------
        parameters = np.zeros(tuple(kwargs.values()), [(x, float) for x in kwargs.keys()])
        grids = np.meshgrid(*[np.linspace(min_max[0], min_max[1], scale)
                              for min_max,scale in zip(args,kwargs.values())], indexing='ij')
        for index, parameter in enumerate(kwargs.keys()):
            parameters[parameter] = grids[index]
        parameters = parameters.reshape(-1)
        return parameters


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


###################################
# -----simulation parameter setting-------
coding_n = 1
MNIST_shape = (1, 784)
coding_duration = 30
duration = coding_duration*MNIST_shape[0]
F_plasticity = 0.2
F_train = 0.05
F_test = 0.05
Dt = defaultclock.dt = 1*ms

#-------class initialization----------------------
function = Function()
base = Base(duration, Dt)
readout = Readout(function.logistic)
result = Result()
MNIST = MNIST_classification(MNIST_shape, duration, Dt)

#-------data initialization----------------------
MNIST.load_Data_MNIST_all(data_path)
df_plasticity = MNIST.select_data(F_plasticity, MNIST.train)
df_train = MNIST.select_data(F_train, MNIST.train)
df_test = MNIST.select_data(F_test, MNIST.test)

df_en_plasticity = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_plasticity, coding_n)
df_en_train = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_train, coding_n)
df_en_test = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_test, coding_n)

data_plasticity_s, label_plasticity = MNIST.get_series_data_list(df_en_plasticity, is_group = True)
data_train_s, label_train = MNIST.get_series_data_list(df_en_train, is_group = True)
data_test_s, label_test = MNIST.get_series_data_list(df_en_test, is_group = True)

#-------get numpy random state------------
np_state = np.random.get_state()


############################################
def grad_search(parameter):
    #---- set numpy random state for each parallel run----
    np.random.set_state(np_state)

    # -----parameter setting-------
    n_ex = 400
    n_inh = int(n_ex/4)
    n_input = MNIST_shape[1]*coding_n
    n_read = n_ex+n_inh

    R = 0.9
    f = 1.2
    f_inE = parameter['f_inE']

    A_EE = 30*f
    A_EI = 60*f
    A_IE = 19*f
    A_II = 19*f
    A_inE = 18*f_inE
    A_inI = 9*f_inE

    p_inE = parameter['p_in']
    p_inI = parameter['p_in']

    learning_rate = 0.001

    #------definition of equation-------------
    neuron_in = '''
    I = stimulus(t,i) : 1
    '''

    neuron = '''
    dv/dt = (I-v) / (15*ms) : 1 (unless refractory)
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

    synapse_stdp = '''
    Switch_plasticity : 1
    w : 1
    w_max : 1
    w_min : 1
    tau_ahead : second
    tau_latter : second
    A_ahead : 1
    A_latter = -A_ahead * tau_ahead / tau_latter * 1.05 : 1
    da_ahead/dt = -a_ahead/tau_ahead : 1 (clock-driven)
    da_latter/dt = -a_latter/tau_latter : 1 (clock-driven)
    '''

    on_pre_ex_stdp = '''
    g+=w
    a_ahead += A_ahead * int(Switch_plasticity)
    w = clip(w+a_latter, w_min, w_max)
    '''

    on_post_ex_stdp = '''
    a_latter += A_latter * int(Switch_plasticity)
    w = clip(w+a_ahead, w_min, w_max)
    '''

    synapse_stdp_inE = '''
    Switch_plasticity : 1
    w_max : 1
    w_min : 1
    w : 1
    mu : 1
    tau_ahead : second
    tau_offset : second
    eta_positive : 1
    eta_negative : 1
    eta_offset : 1
    doffset/dt = -offset/tau_offset : 1 (clock-driven)
    da_ahead/dt = -a_ahead/tau_ahead : 1 (clock-driven)
    '''

    on_pre_ex_stdp_inE = '''
    g+=w
    a_ahead +=  1
    w = clip(w + int(Switch_plasticity) * (a_ahead - offset) * eta_positive * (w_max-w_min)* (w_max-w)/(w_max-w_min), w_min, w_max)
    '''

    on_post_ex_stdp_inE = '''
    offset = offset + (1-offset) * eta_offset
    w = clip(w - int(Switch_plasticity) * eta_negative* (w_max-w_min) * (1 - mu*(w-w_min)/(w_max-w_min)), w_min, w_max)
    '''

    # -----Neurons and Synapses setting-------
    Input = NeuronGroup(n_input, neuron_in, threshold='I > 0', method='euler', refractory=0 * ms,
                        name = 'neurongroup_input')

    G_ex = NeuronGroup(n_ex, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                    name ='neurongroup_ex')

    G_inh = NeuronGroup(n_inh, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                    name ='neurongroup_in')

    G_readout = NeuronGroup(n_read, neuron_read, method='euler', name='neurongroup_read')

    S_inE = Synapses(Input, G_ex, synapse_stdp_inE, on_pre=on_pre_ex_stdp_inE, on_post=on_post_ex_stdp_inE,
                     method='euler', name='synapses_inE')

    S_inI = Synapses(Input, G_inh, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_inI')

    S_EE = Synapses(G_ex, G_ex, synapse_stdp, on_pre = on_pre_ex_stdp, on_post= on_post_ex_stdp,method='euler', name='synapses_EE')

    S_EI = Synapses(G_ex, G_inh, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_EI')

    S_IE = Synapses(G_inh, G_ex, synapse, on_pre = on_pre_inh ,method='euler', name='synapses_IE')

    S_II = Synapses(G_inh, G_inh, synapse, on_pre = on_pre_inh ,method='euler', name='synapses_I')

    S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_ex, method='euler')

    S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_inh, method='euler')

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

    [G_ex,G_in] = base.allocate([G_ex,G_inh],5,5,20)

    # -------initialization of network topology and synapses parameters----------
    S_inE.connect(condition='j<0.3*N_post', p = p_inE)
    S_inI.connect(condition='j<0.3*N_post', p = p_inI)
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

    S_EE.pre.delay = '1.5*ms'
    S_EI.pre.delay = '0.8*ms'
    S_IE.pre.delay = '0.8*ms'
    S_II.pre.delay = '0.8*ms'

    S_EE.w_max = np.max(S_EE.w)
    S_EE.w_min = np.min(S_EE.w)
    S_EE.A_ahead = learning_rate * (np.max(S_EE.w) - np.min(S_EE.w))
    base.set_local_parameter(S_EE, 'tau_ahead', (0.005, 0.1), method='group', group_n=10, location_label='z_post')
    base.set_local_parameter(S_EE, 'tau_latter', (0.005, 0.1), method='group', group_n=10, location_label='z_post')

    S_inE.w_max = np.max(S_inE.w)
    S_inE.w_min = np.min(S_inE.w)
    S_inE.mu = 0.9
    S_inE.offset = 0.0
    S_inE.tau_ahead = '3*ms'
    S_inE.tau_offset = '20*ms'
    S_inE.eta_offset = 0.5
    base.set_local_parameter(S_inE, 'eta_positive', (0.3, 0.9))
    base.set_local_parameter(S_inE, 'eta_negative', (0.01, 0.1))

    # ------create network-------------
    net = Network(collect())
    net.store('init')

    # ---- define network run function----
    def run_net(inputs):
        states = None
        for ser, data in enumerate(inputs):
            if ser % 100 == 0:
                print('The simulation is running at %s time.' % ser)
            stimulus = TimedArray(data, dt=Dt)
            net.run(duration * Dt)
            states = base.np_append(states, G_readout.variables['v'].get_value())
            net.restore('init')
        return (MinMaxScaler().fit_transform(states)).T

    def run_net_plasticity(inputs, *args, **kwargs):
        metric_plasticity = {
            'weight_changed': None}
        metric_plasticity_list = [metric_plasticity for S in args]
        for ser, data in enumerate(inputs):
            weight_initial = [S.variables['w'].get_value().copy() for S in args]
            if ser % 50 == 0:
                print('The simulation is running at %s time' % ser)
            stimulus = TimedArray(data, dt=Dt)
            net.run(duration * Dt)
            weight_trained = [S.variables['w'].get_value().copy() for S in args]
            metric_plasticity_list = [base.update_metrics('numpy', x - y, **metric)
                                 for x, y, metric in
                                 zip(weight_trained, weight_initial, metric_plasticity_list)]
            net.restore('init')
            for S_index, S in enumerate(args):
                S.w = weight_trained[S_index].copy()
        confusion = base.get_confusion(base.get_plasticity_confuse(metric_plasticity_list, kwargs['label']))
        return confusion

    def run_net_plasticity_by_category(inputs, *args, **kwargs):
        def allocate_post_neuron_to_label(S, label):
            dfs = []
            for s in S:
                N_n, N_l = len(np.unique(s.j)), len(np.unique(label))
                N_per_group = N_n // N_l
                Group = [np.unique(s.j)[i * N_per_group:(i + 1) * N_per_group] for i in range(N_l)]
                if N_n % N_l != 0:
                    Group.append(np.unique(s.j)[N_per_group * N_l:])
                df = pd.DataFrame({'neuron': pd.Series(Group), 'label': pd.Series(np.unique(label))})
                dfs.append(df)
            return dfs

        metric_plasticity = {
            'weight_changed': None,}
        metric_plasticity_list = [metric_plasticity for S in args]
        Groups = allocate_post_neuron_to_label(kwargs['local_synapse'], kwargs['label'])
        for ser, (data, l) in enumerate(zip(inputs, kwargs['label'])):
            for index, G in enumerate(Groups):
                Switch = np.zeros(args[index].N_post)
                Switch[G[G['label'] == l]['neuron'].tolist()[0]] = 1
                kwargs['local_synapse'][index].variables['Switch_plasticity'].set_value(
                    Switch[kwargs['local_synapse'][index].j])
            weight_initial = [S.variables['w'].get_value().copy() for S in args]
            if ser % 50 == 0:
                print('The simulation is running at %s time' % ser)
            stimulus = TimedArray(data, dt=Dt)
            net.run(duration * Dt)
            weight_trained = [S.variables['w'].get_value().copy() for S in args]
            metric_plasticity_list = [base.update_metrics('numpy', x - y, **metric)
                                          for x, y, metric in
                                          zip(weight_trained, weight_initial, metric_plasticity_list)]
            net.restore('init')
            for S_index, S in enumerate(args):
                S.w = weight_trained[S_index].copy()
        confusion = base.get_confusion(base.get_plasticity_confuse(metric_plasticity_list, kwargs['label']))
        return confusion

    def run_net_plasticity_by_local_randomly(inputs, *args, **kwargs):
        metric_plasticity = {
            'weight_changed': None}
        metric_plasticity_list = [metric_plasticity for S in args]
        for ser, (data, l) in enumerate(zip(inputs, kwargs['label'])):
            for index, S in enumerate(args):
                temp = np.unique(S.j)
                np.random.shuffle(temp)
                Group = temp[:int(len(temp) * kwargs['fraction'])]
                Switch = np.zeros(args[index].N_post)
                Switch[Group[Group['label'] == l]['neuron']] = 1
                args[index].variables['Switch_plasticity'].set_value(Switch[args[index].j])
            weight_initial = [S.variables['w'].get_value().copy() for S in args]
            if ser % 50 == 0:
                print('The simulation is running at %s time' % ser)
            stimulus = TimedArray(data, dt=Dt)
            net.run(duration * Dt)
            weight_trained = [S.variables['w'].get_value().copy() for S in args]
            metric_plasticity_list = [base.update_metrics('numpy', x - y, **metric)
                                          for x, y, metric in
                                          zip(weight_trained, weight_initial, metric_plasticity_list)]
            net.restore('init')
            for S_index, S in enumerate(args):
                S.w = weight_trained[S_index].copy()
        confusion = base.get_confusion(base.get_plasticity_confuse(metric_plasticity_list, kwargs['label']))
        return confusion

    # --------open plasticity--------
    S_EE.Switch_plasticity = True
    S_inE.Switch_plasticity = True
    net._stored_state['init'][S_EE.name]['Switch_plasticity'] = S_EE._full_state()['Switch_plasticity']
    net._stored_state['init'][S_inE.name]['Switch_plasticity'] = S_inE._full_state()['Switch_plasticity']

    # ------run for plasticity-------
    confusion = run_net_plasticity_by_category(data_plasticity_s, S_EE, S_inE,
                                                label= label_plasticity, local_synapse =[S_inE])

    # confusion = run_net_plasticity_by_local_randomly(data_plasticity_s, S_EE, S_inE,
    #                                                  label=label_plasticity, local_synapse=[S_inE], fraction=0.1)

    #-------close plasticity--------
    S_EE.Switch_plasticity = False
    S_inE.Switch_plasticity = False
    net._stored_state['init'][S_EE.name]['w'] = S_EE._full_state()['w']
    net._stored_state['init'][S_inE.name]['w'] = S_inE._full_state()['w']
    net._stored_state['init'][S_EE.name]['Switch_plasticity'] = S_EE._full_state()['Switch_plasticity']
    net._stored_state['init'][S_inE.name]['Switch_plasticity'] = S_inE._full_state()['Switch_plasticity']

    # ------run for train-------
    states_train = run_net(data_train_s)

    # ------run for test--------
    states_test = run_net(data_test_s)

    # ------Readout---------------
    score_train, score_test = readout.readout_sk(states_train, states_test, label_train, label_test, solver="lbfgs",
                                                 multi_class="multinomial")

    # ----------show results-----------
    print('parameters %s' % parameter)
    print('confusion: ', confusion)
    print('Train score: ', score_train)
    print('Test score: ', score_test)

    return np.array([(confusion, score_train, score_test, parameter)],
                    [('confusion', object), ('score_train', float), ('score_test', float), ('parameters', object)])


##########################################
# -------prepare parameters---------------
if __name__ == '__main__':
    core = 10
    pool = Pool(core)
    parameters = base.parameters_GS((0.01, 0.1), (0.1, 0.5), f_inE=10, p_in=10)

    # -------parallel run---------------
    score = np.asarray(pool.map(grad_search, parameters))

    # --------get the final results-----
    score = np.asarray(score)
    highest_score_test = np.max(score['score_test'])
    score_train = score['score_train'][np.where(score['score_test'] == highest_score_test)[0]]
    best_parameter = score['parameters'][np.where(score['score_test'] == highest_score_test)[0]]
    confusion = score['confusion'][np.where(score['score_test'] == highest_score_test)[0]]

    # --------show the final results-----
    print('highest_score_test is %s, score_train is %s, best_parameter is %s, confusion is %s'
          % (highest_score_test, score_train, best_parameter, confusion))

    # -----------save the final results-------
    result.result_save('score_grid_search_STDP.pkl', score=score)
    result.result_save('best_parameters_STDP.pkl', best_parameter=best_parameter)
