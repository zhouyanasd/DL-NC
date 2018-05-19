# ----------------------------------------
# using logistic regression
# change the LSM structure according to W.Maass paper
# ST task in W.Maass paper
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


# ------define general function------------
class Function():
    def __init__(self):
        pass

    def logistic(self, f):
        return 1/(1+np.exp(-f))

    def softmax(self, z):
        return np.array([(np.exp(i) / np.sum(np.exp(i))) for i in z])

    def gamma(self, a, size):
        return sp.stats.gamma.rvs(a, size = size)


class Base():
    def __init__(self, duration, dt):
        self.duration = duration
        self.dt = dt
        self.interval = duration*dt

    def get_states(self, input, running_time, sample, normalize = False):
        n = int(running_time / self.interval)
        step = int(self.interval / sample / defaultclock.dt)
        interval_ = int(self.interval / defaultclock.dt)
        temp = []
        for i in range(n):
            sum = np.sum(input[:, i * interval_: (i + 1) * interval_][:,::-step], axis=1)
            temp.append(sum)
        if normalize:
            return MinMaxScaler().fit_transform(np.asarray(temp)).T
        else:
            return np.asarray(temp).T

    def update_states(self, type = 'pandas', *args, **kwargs):
        for seq, state in enumerate(kwargs):
            if type == 'pandas':
                kwargs[state] = kwargs[state].append(pd.DataFrame(args[seq]))
            elif type == 'numpy':
                kwargs[state] = self.np_extend(kwargs[state],args[seq],1)
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
        V = np.zeros((X,Y,Z), [('x',float),('y',float),('z',float)])
        V['x'], V['y'], V['z'] = np.meshgrid(np.linspace(0,X-1,X),np.linspace(0,X-1,X),np.linspace(0,Z-1,Z))
        V = V.reshape(X*Y*Z)
        np.random.shuffle(V)
        n = 0
        for g in G:
            for i in range(g.N):
                g.x[i], g.y[i], g.z[i]= V[n][0], V[n][1], V[n][2]
                n +=1
        return G


    def w_norm2(self, n_post, Synapsis):
        for i in range(n_post):
            a = Synapsis.w[np.where(Synapsis._synaptic_post == i)[0]]
            Synapsis.w[np.where(Synapsis._synaptic_post == i)[0]] = a/np.linalg.norm(a)

    def np_extend(self, a, b, axis = 0):
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
        return np.append(a, b.reshape(tuple(shape)), axis= 0)


class Readout():
    def __init__(self, function):
        self.function = function

    def data_init(self, M_train, M_test, label_train, label_test, rate,theta):
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

    def predict_logistic(self,results):
        labels = (results > 0.5).astype(int).T
        return labels

    def calculate_score(self,label, label_predict):
        return [accuracy_score(i,j) for i,j in zip(label,label_predict)]

    def add_bis(self,data):
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
        return ((self.cost_train - self.cost(self.X_train, self.Y_train, self.P))> self.theta).any() and \
               ((self.cost_test - self.cost(self.X_test, self.Y_test, self.P)) > self.theta).any() or self.iter < 100

    def readout(self):
        self.iter = 0
        while self.stop_condition():
            self.iter += 1
            self.cost_train = self.cost(self.X_train, self.Y_train, self.P)
            self.cost_test = self.cost(self.X_test, self.Y_test, self.P)
            self.P = self.train(self.X_train, self.Y_train, self.P)
            if self.iter %10000 == 0:
                print(self.iter, self.cost_train, self.cost_test)
        print(self.iter, self.cost_train, self.cost_test)
        return self.test(self.X_train, self.P), self.test(self.X_test, self.P)

    def readout_sk(self, X_train, X_test, y_train, y_test):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(X_train.T, y_train.T)
        y_train_predictions = lr.predict(X_train.T)
        y_test_predictions = lr.predict(X_test.T)
        return accuracy_score(y_train_predictions, y_train.T), accuracy_score(y_test_predictions, y_test.T)


class ST_classification_mass():
    def __init__(self, n_p, n, duration, frequency, dt):
        self.n_p = n_p
        self.n = n
        self.duration = duration
        self.frequency = frequency
        self.dt = dt
        self.n_1 = int(ceil(duration *dt * frequency))
        self.n_0 = int(ceil(duration)) - self.n_1
        self.D = int(duration / n)
        self.pattern_generation()

    def pattern_generation(self):
        self.pattern = []
        for i in range(self.n_p):
            b = [1] * self.n_1
            b.extend([0] * self.n_0)
            c = np.array(b)
            np.random.shuffle(c)
            self.pattern.append(c)

    def data_generation(self):
        value = []
        label = []
        for i in range(self.n):
            select = np.random.choice(self.n_p)
            fragment = np.random.choice(self.n)
            value.extend(self.pattern[select][self.D * fragment:self.D * (fragment + 1)])
            label.append(select)
            data = {'label': label, 'value': value}
        return data

    def data_noise_jitter(self, data):
        pass

    def data_generation_batch(self, number, noise = False):
        value = []
        label = []
        for i in range(number):
            data = self.data_generation()
            if noise :
                self.data_noise_jitter(data)
            value.append(data['value'])
            label.append(data['label'])
            df = pd.DataFrame({'value': pd.Series(value), 'label': pd.Series(label)})
        return df

    def get_series_data(self, data_frame, is_order=True, is_group=False, *args, **kwargs):
        if not is_order:
            data_frame_obj = data_frame.sample(frac=1).reset_index(drop=True)
        else:
            data_frame_obj = data_frame
        data_frame_s = []
        if not is_group:
            for value in data_frame_obj['value']:
                data_frame_s.extend(np.asarray([value]).T)
        else:
            for value in data_frame_obj['value']:
                data_frame_s.append(np.asarray([value]).T)
        label = data_frame_obj['label']
        return np.asarray(data_frame_s), np.array(list(map(list, zip(*label))))


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

    def animation(self, t, v, interval, duration, a_step=10, a_interval=100, a_duration = 10):
        xs = LinearScale()
        ys = LinearScale()
        line = Lines(x=t[:interval], y=v[:,:interval], scales={'x': xs, 'y': ys})
        xax = Axis(scale=xs, label='x', grid_lines='solid')
        yax = Axis(scale=ys, orientation='vertical', tick_format='0.2f', label='y', grid_lines='solid')
        fig = Figure(marks=[line], axes=[xax, yax], animation_duration=a_duration)
        def on_value_change(change):
            line.x = t[change['new']:interval+change['new']]
            line.y = v[:,change['new']:interval+change['new']]
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


# #--------define network run function-------------------
def run_net(inputs, record = True):
    states = None
    monitor_record= {
        'm_g_ex.r': None,
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
        if record :
            monitor_record= base.update_states('numpy', m_g_ex.r, m_g_ex.I, m_g_ex.v, m_g_in.I, m_g_in.v, m_read.I,
                                                m_read.v, m_input.I, **monitor_record)
        net.restore('init')
    return (MinMaxScaler().fit_transform(states)).T, monitor_record


###################################
# -----parameter setting-------
duration = 1000
N_train = 1000
N_test = 500
Dt = defaultclock.dt = 1*ms

sample = 1
train_rate = 1e-4
train_theta = 1e-6
pre_train_loop = 0

n_ex = 108
n_inh = int(n_ex/4)
n_input = 1
n_read = n_ex+n_inh

R = 2

A_EE = 30
A_EI = 60
A_IE = 19
A_II = 19
A_inE = 18
A_inI = 9

adr = 1.5
###########################################
#-------class initialization----------------------
function = Function()
base = Base(duration, Dt)
readout = Readout(function.logistic)
result = Result()
ST = ST_classification_mass(2, 4, duration, 20*Hz, Dt)

#-------data initialization----------------------
df_train = ST.data_generation_batch(N_train)
df_test = ST.data_generation_batch(N_test)

data_train_s, label_train = ST.get_series_data(df_train, is_group = True)
data_test_s, label_test = ST.get_series_data(df_test, is_group = True)

#------definition of equation-------------
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
dr/dt = (15-r)/(20*ms) : 1
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

threshold_reset = 'v > r'

adaptive_reset = '''
v = 13.5
r = r+adr
'''

# -----Neurons and Synapses setting-------
Input = NeuronGroup(n_input, neuron_in, threshold='I > 0', method='euler', refractory=0 * ms,
                    name = 'neurongroup_input')

G_ex = NeuronGroup(n_ex, neuron, threshold=threshold_reset, reset=adaptive_reset, method='euler', refractory=2.99 * ms,
                name ='neurongroup_ex')

G_inh = NeuronGroup(n_inh, neuron, threshold=threshold_reset, reset=adaptive_reset, method='euler', refractory=1.99 * ms,
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
G_ex.r = '15'
G_inh.r = '15'
G_ex.v = '13.5+1.5*rand()'
G_inh.v = '13.5+1.5*rand()'
G_readout.v = '0'
G_ex.g = '0'
G_inh.g = '0'
G_readout.g = '0'
G_ex.h = '0'
G_inh.h = '0'
G_readout.h = '0'

[G_ex,G_in] = base.allocate([G_ex,G_inh],3,3,15)


# -------initialization of network topology and synapses parameters----------
S_inE.connect(condition='j<0.3*N_post')
S_inI.connect(condition='j<0.3*N_post')
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
m_g_ex = StateMonitor(G_ex, (['I', 'v', 'r']), record=True)
m_g_in = StateMonitor(G_in, (['I', 'v']), record=True)
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
score_train = []
score_test = []
for i in range(label_train.shape[0]):
    readout.data_init(states_train, states_test, label_train[i:i+1], label_test[i:i+1], train_rate, train_theta)
    Y_train_, Y_test_ = readout.readout()
    label_train_ = readout.predict_logistic(Y_train_)
    label_test_ = readout.predict_logistic(Y_test_)
    score_train.extend(readout.calculate_score(label_train[i:i+1],label_train_))
    score_test.extend(readout.calculate_score(label_test[i:i+1], label_test_))


#####################################
#----------show results-----------
print('Train score: ',score_train)
print('Test score: ',score_test)


#####################################
#-----------save monitor data-------
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
play, slider, fig = result.animation(np.arange(monitor['m_read.v'].shape[1]), monitor['m_read.v'], 100, N_test*duration)
widgets.VBox([widgets.HBox([play, slider]),fig])