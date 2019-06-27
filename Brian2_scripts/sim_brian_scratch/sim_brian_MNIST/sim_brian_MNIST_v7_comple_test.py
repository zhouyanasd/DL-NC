# ----------------------------------------
# Load CMA-ES_v1.1, BO_v3 and SAES_v1 results
# run for another test 1000 data
# ----------------------------------------

from brian2 import *
import scipy as sp
import struct
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from bqplot import *
import warnings
import re
import os
import time
from multiprocessing import Pool
import cma
from functools import partial


warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)
data_path = '../../../Data/MNIST_data/'


# ------define general function------------
class timelog():
    def __init__(self, func):
        self.func = func
        self.itime = time.time()
        self.iteration = 0
        with open('Results_Record' + '.dat', 'w') as f:
            f.write('iteration' + ' '
                    + 'wall_time' + ' '
                    + 'result_validation' + ' '
                    + 'result_test' + ' '
                    + 'parameters' + ' '
                    + '\n')

    def __call__(self, *args, **kwargs):
        validation, test, parameters= self.func(*args, **kwargs)
        self.save(validation, test, parameters)
        return validation

    @property
    def elapsed(self):
        return time.time() - self.itime

    def save(self, validation, test, parameters):
        self.iteration += 1
        with open('Results_Record' + '.dat', 'a') as f:
            f.write(str(self.iteration) + ' ' + str(self.elapsed) + ' ' + str(validation) + ' '
                    + str(test) + ' '+ str(parameters) + ' ' + '\n')

class Complement_test():
    def __init__(self, type, path):
        self.type = type
        self.path = path

    def load_SAES(self):
        X,fit =[],[]
        with open(self.path, 'r') as f:
            l = f.readlines()
        l.pop(0)
        p1 = re.compile(r'[{](.*?)[}]', re.S)
        for i in range(0, len(l)):
            l[i] = l[i].rstrip('\n')
            s = re.findall(p1, l[i])[0]
            d = eval('{'+s+'}')
            X.append(d)
            f = float(l[i].replace('{'+s+'}','').split(' ')[2])
            fit.append(f)
        return X, fit

    def load_CMAES(self):
        fit = np.loadtext(self.path, delimiter=None, comments='%', usecols=(4))
        X = np.loadtext(self.path, delimiter=None, comments='%', usecols=(5,6,7,8,9,10,11,12,13))
        return X, fit

    def load_BO(self):
        X, fit = [], []
        with open(self.path, 'r') as load_f:
            while True:
                try:
                    iteration = next(load_f)
                except StopIteration:
                    break
                iteration = json.loads(iteration)
                X.append(iteration['params'])
                fit.append(iteration['target'])
        return X, fit

    def load(self):
        if self.type == 'CMA-ES':
            self.X, self.res = self.load_CMAES()
        elif self.type == 'SAES':
            self.X, self.res = self.load_SAES()
        elif self.type == 'BO':
            self.X, self.res =  self.load_BO()
        else:
            print('wrong type')

    def run(self, f):
        for parameter, validation in zip(self.X, self.res):
            f(validation, parameter)

class Base():
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


class Readout():
    def readout_sk(self, X_train, X_test, y_train, y_test, **kwargs):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(**kwargs)
        lr.fit(X_train.T, y_train.T)
        y_train_predictions = lr.predict(X_train.T)
        y_test_predictions = lr.predict(X_test.T)
        return accuracy_score(y_train_predictions, y_train.T), accuracy_score(y_test_predictions, y_test.T)


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
TYPE = 'CMA-ES' # or 'BO' 'SAES'

# -----simulation parameter setting-------
coding_n = 3
MNIST_shape = (1, 784)
coding_duration = 30
duration = coding_duration*MNIST_shape[0]
F_train = 0.05
F_validation = 0.05
F_test = 0.05
Dt = defaultclock.dt = 1*ms

#-------class initialization----------------------
complement = Complement_test(TYPE,path='')
base = Base(duration, Dt)
readout = Readout(function.logistic)
MNIST = MNIST_classification(MNIST_shape, duration, Dt)

#-------data initialization----------------------
MNIST.load_Data_MNIST_all(data_path)
df_train = MNIST.select_data(F_train, MNIST.train)
df_validation = MNIST.select_data(F_validation, MNIST.test)
df_test = MNIST.select_data(F_test, MNIST.test)

df_en_train = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_train, coding_n)
df_en_validation = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_validation, coding_n)
df_en_test = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_test, coding_n)

data_train_s, label_train = MNIST.get_series_data_list(df_en_train, is_group = True)
data_validation_s, label_validation = MNIST.get_series_data_list(df_en_validation, is_group = True)
data_test_s, label_test = MNIST.get_series_data_list(df_en_test, is_group = True)

#-------get numpy random state------------
np_state = np.random.get_state()


############################################
# ---- define network run function----
def run_net(inputs, parameter):

    """
        run_net(inputs, parameter)
            Parameters = [R, p_inE/I, f_in, f_EE, f_EI, f_IE, f_II, tau_ex, tau_inh]
            ----------
    """

    #---- set numpy random state for each run----
    np.random.set_state(np_state)
    if type(parameter) == dict:
        parameter = np.array(list(parameter.values))

    # -----parameter setting-------
    n_ex = 1600
    n_inh = int(n_ex/4)
    n_input = MNIST_shape[1]*coding_n
    n_read = n_ex+n_inh

    R = parameter[0]
    f_in = parameter[2]
    f_EE = parameter[3]
    f_EI = parameter[4]
    f_IE = parameter[5]
    f_II = parameter[6]

    A_EE = 60*f_EE
    A_EI = 60*f_EI
    A_IE = 60*f_IE
    A_II = 60*f_II
    A_inE = 60*f_in
    A_inI = 60*f_in

    tau_ex = parameter[7]*coding_duration
    tau_inh = parameter[8]*coding_duration
    tau_read= 30

    p_inE = parameter[1]*0.1
    p_inI = parameter[1]*0.1

    #------definition of equation-------------
    neuron_in = '''
    I = stimulus(t,i) : 1
    '''

    neuron = '''
    tau : 1
    dv/dt = (I-v) / (tau*ms) : 1 (unless refractory)
    dg/dt = (-g)/(3*ms) : 1
    dh/dt = (-h)/(6*ms) : 1
    I = (g+h)+13.5: 1
    x : 1
    y : 1
    z : 1
    '''

    neuron_read = '''
    tau : 1
    dv/dt = (I-v) / (tau*ms) : 1
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
    G_ex.tau = tau_ex
    G_inh.tau = tau_inh
    G_readout.tau = tau_read
    [G_ex,G_in] = base.allocate([G_ex,G_inh],10,10,20)


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

    # ------create network-------------
    net = Network(collect())
    net.store('init')

    # ------run network-------------
    stimulus = TimedArray(inputs[0], dt=Dt)
    net.run(duration * Dt)
    states = net.get_states()['neurongroup_read']['v']
    net.restore('init')
    return (states, inputs[1])

@timelog
def parameters_search(validation, parameter):
    # ------parallel run for train-------
    states_train_list = pool.map(partial(run_net, parameter = parameter), [(x) for x in zip(data_train_s, label_train)])
    # ----parallel run for test--------
    states_test_list = pool.map(partial(run_net, parameter = parameter), [(x) for x in zip(data_test_s, label_test)])
    # ------Readout---------------
    states_train, states_test, _label_train, _label_test = [], [], [], []
    for train in states_train_list :
        states_train.append(train[0])
        _label_train.append(train[1])
    for test in states_test_list:
        states_test.append(test[0])
        _label_test.append(test[1])
    states_train = (MinMaxScaler().fit_transform(np.asarray(states_train))).T
    states_test = (MinMaxScaler().fit_transform(np.asarray(states_test))).T
    score_train, score_test = readout.readout_sk(states_train, states_test,
                                                 np.asarray(_label_train), np.asarray(_label_test),
                                                 solver="lbfgs", multi_class="multinomial")
    # ----------show results-----------
    print('parameters %s' % parameter)
    print('Train score: ', score_train)
    print('Test score: ', score_test)
    return validation, 1 - score_test, parameter

##########################################
# -------CMA-ES parameters search---------------
if __name__ == '__main__':
    core = 10
    pool = Pool(core)
    complement.run(parameters_search)