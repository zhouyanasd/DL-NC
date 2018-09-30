# ----------------------------------------
# LSM with BCM for MNIST test
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


warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)
data_path = '../../../Data/MNIST_data/'


#-------define brian2 function------------
@check_units(spike_window=1,result=1)
def get_rate(spike_window):
    return np.sum(spike_window, axis = 1)/spike_window.shape[1]

@check_units(spike_window=1, spike = 1, result=1)
def get_spike_window(spike_window, spike):
    new_window = np.zeros(spike_window.shape)
    new_window[:,:-1] = spike_window[:,1:]
    new_window[:,-1] = spike
    return new_window


#-------define general function------------
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

    def spectral_radius(self, S):
        if isinstance(S, Synapses):
            n_pre = S.N_pre
            n_post = S.N_post
            sources = S.i[:]
            targets = S.j[:]
            values = S.w[:] - np.mean(S.variables['w'].get_value())
            if sources.shape[0] == targets.shape[0]:
                ma = self.connection_matrix(n_pre, n_post, sources, targets, values) / np.sqrt(sources.shape[0])
            else:
                raise ('Only synapses with the same source and target can calculate spectral radius')
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

    n_ex = 400
    n_inh = int(n_ex/4)
    n_input = MNIST_shape[1]*coding_n
    n_read = n_ex+n_inh

    R = 0.9
    f = 1.2

    A_EE = 30*f
    A_EI = 60*f
    A_IE = 19*f
    A_II = 19*f
    A_inE = 18*f
    A_inI = 9*f

    p_inE = 0.01
    p_inI = 0.01

    rate_window = 20

    #------definition of equation-------------
    neuron_in = '''
    I = stimulus(t,i) : 1
    '''

    neuron_ex = '''
    rate : 1
    spike : 1
    dv/dt = (I-v) / (15*ms) : 1 (unless refractory)
    dg/dt = (-g)/(3*ms) : 1
    dh/dt = (-h)/(6*ms) : 1
    I = (g+h)+13.5: 1
    x : 1
    y : 1
    z : 1
    '''

    neuron_inh = '''
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

    reset_ex= '''
    v = 0
    spike = 1
    '''

    event_ex = '''
    spike_window = get_spike_window(spike_window, spike)
    rate = get_rate(spike_window)
    spike = 0
    '''

    synapse = '''
    w : 1
    '''

    synapse_bcm = '''
    Switch_plasticity : 1
    weight_decay : 1
    A_bcm : 1
    w : 1
    w_max : 1
    w_min : 1
    tau : second
    dth_m/dt = (rate_post - th_m)/tau : 1 (clock-driven)
    '''

    on_pre_ex = '''
    g+=w
    '''

    on_pre_inh = '''
    h-=w
    '''

    on_pre_ex_bcm = {
        'pre': '''
         g += w
        ''',
        'pathway_rate':'''
         d_w = A_bcm*rate_pre*(rate_post - th_m)*rate_post - weight_decay*w
         w = clip(w + d_w * int(Switch_plasticity) , w_min, w_max)
        '''}

    # -----Neurons and Synapses setting-------
    Input = NeuronGroup(n_input, neuron_in, threshold='I > 0', method='euler', refractory=0 * ms,
                        name = 'neurongroup_input')

    G_ex = NeuronGroup(n_ex, neuron_ex, threshold='v > 15', reset = reset_ex, method='euler', refractory=3 * ms,
                     events={'event_rate':'True'}, name ='neurongroup_ex')

    G_inh = NeuronGroup(n_inh, neuron_inh, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                    name ='neurongroup_in')

    G_readout = NeuronGroup(n_read, neuron_read, method='euler', name='neurongroup_read')

    S_inE = Synapses(Input, G_ex, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_inE')

    S_inI = Synapses(Input, G_inh, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_inI')

    S_EE = Synapses(G_ex, G_ex, synapse_bcm, on_pre = on_pre_ex_bcm, on_event={'pre':'spike', 'pathway_rate': 'event_rate'},
                    method='euler', name='synapses_EE')

    S_EI = Synapses(G_ex, G_inh, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_EI')

    S_IE = Synapses(G_inh, G_ex, synapse, on_pre = on_pre_inh ,method='euler', name='synapses_IE')

    S_II = Synapses(G_inh, G_inh, synapse, on_pre = on_pre_inh ,method='euler', name='synapses_I')

    S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_ex, method='euler')

    S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_inh, method='euler')

    #-------initialization of neuron parameters----------
    G_ex.run_on_event('event_rate', event_ex)
    G_ex.variables.add_dynamic_array('spike_window', size=(n_ex,rate_window))
    G_ex.rate = 0
    G_ex.spike = 0
    G_ex.spike_window = 0

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
    S_EE.tau = parameter['tau']*ms
    S_EE.A_bcm = parameter['A_bcm']
    S_EE.weight_decay = 0.00001

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

    # --------open plasticity--------
    S_EE.Switch_plasticity = True
    net._stored_state['init'][S_EE.name]['Switch_plasticity'] = S_EE._full_state()['Switch_plasticity']

    # ------run for plasticity-------
    confusion = run_net_plasticity(data_plasticity_s, S_EE, label= label_plasticity)

    #-------close plasticity--------
    S_EE.Switch_plasticity = False
    net._stored_state['init'][S_EE.name]['w'] = S_EE._full_state()['w']
    net._stored_state['init'][S_EE.name]['Switch_plasticity'] = S_EE._full_state()['Switch_plasticity']

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
    parameters = base.parameters_GS((100, 500), (2, 20), A_bcm = 10, tau=10)

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
    result.result_save('score_grid_search_BCM.pkl', score=score)
    result.result_save('best_parameters_BCM.pkl', best_parameter=best_parameter)


