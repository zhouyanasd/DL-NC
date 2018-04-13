# ----------------------------------------
# LSM without STDP for MNIST test
# add neurons to readout layer for multi-classification(one-versus-the-rest)
# using softmax(logistic regression)
# input layer is changed to 781*1 with encoding method
# change the LSM structure according to Maass paper
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


prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)

# ------define function------------
class Function():
    def __init__(self):
        pass

    def logistic(self, f):
        return 1/(1+np.exp(-f))


    def softmax(self, z):
        return np.array([(np.exp(i) / np.sum(np.exp(i))) for i in z])


class Base():
    def __init__(self, duration, dt):
        self.duration = duration
        self.dt = dt
        self.interval = duration*dt

    def get_states(self, input, running_time, sample):
        n = int(running_time / self.interval)
        step = int(self.interval / sample / defaultclock.dt)
        interval_ = int(self.interval / defaultclock.dt)
        temp = []
        for i in range(n):
            sum = np.sum(input[:, i * interval_: (i + 1) * interval_][:,::-step], axis=1)
            temp.append(sum)
        return MinMaxScaler().fit_transform(np.asarray(temp).T)

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


class Readout():
    def __init__(self, function):
        self.function = function

    def cost(self, X, Y, P):
        left = np.multiply(Y, np.log(self.function(X.dot(P))))
        right = np.multiply((1 - Y), np.log(1 - self.function(X.dot(P))))
        return -np.sum((left + right), axis=0) / (len(Y))

    def train(self,X, Y, P, rate=0.0001, theta=1e-2):
        time = 0
        while (self.cost(X, Y, P) > theta).all():
            time += 1
            P = P + X.T.dot(Y - self.function(X.dot(P))) * rate
        print(time, self.cost(X, Y, P))
        return P

    def predict_logistic(self,results):
        labels = (results > 0.5).astype(int).T
        return labels

    def prepare_Y(self, label):
        if np.asarray(label).ndim == 1:
            return np.asarray([label])
        else:
            return np.asarray(label)

    def lms_test(self, Data, p):
        one = np.ones((Data.shape[1], 1))  # bis
        X = np.hstack((Data.T, one))
        return self.function(X.dot(p))

    def readout(self, M, Y):
        one = np.ones((M.shape[1], 1))
        X = np.hstack((M.T, one))
        P = np.random.rand(X.shape[1], Y.T.shape[1])
        para = self.train(X, Y.T, P)
        return para

    def calculate_score(self,label, label_predict):
        return [accuracy_score(i,j) for i,j in zip(label,label_predict)]


class ST_classification_mass():
    def __init__(self, n_p, n, duration, frequency, dt):
        self.n_p = n_p
        self.n = n
        self.duration = duration
        self.frequency = frequency
        self.dt = dt
        self.n_1 = int(ceil(duration * frequency))
        self.n_0 = int(ceil(duration / dt)) - self.n_1
        self.D = int(duration / n / dt)
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

    def get_series_data(self, data_frame, is_order=True, *args, **kwargs):
        if not is_order:
            data_frame_obj = data_frame.sample(frac=1).reset_index(drop=True)
        else:
            data_frame_obj = data_frame
        data_frame_s = []
        for value in data_frame_obj['value']:
            data_frame_s.extend(value)
        data_frame_s = np.asarray(data_frame_s)
        label = data_frame_obj['label']
        return data_frame_s, list(map(list, zip(*label)))


class Result():
    def __init__(self):
        pass

    def result_save(self, path, *arg, **kwarg):
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


# -----parameter setting-------
duration = 1000
N_train = 1000
N_test = 500
Dt = defaultclock.dt = 1*ms
sample = 1
pre_train_loop = 0

n_ex = 108
n_inh = int(n_ex/4)
n_input = 1
n_read = n_ex+n_inh

R = 2

A_EE = 30
A_EI = 60
A_IE = -19
A_II = -19
A_inE = 18
A_inI = 9


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

data_train_s, label_train = ST.get_series_data(df_train, False)
data_test_s, label_test = ST.get_series_data(df_test, False)

duration_train = len(data_train_s) * Dt
duration_test = len(data_test_s) * Dt

Time_array_train = TimedArray(data_train_s, dt=Dt)
Time_array_test = TimedArray(data_test_s, dt=Dt)

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

synapse_dynamic = '''
du/dt = (-u)/(F*second) : 1 (event-driven)
dr/dt = (-r)/(D*second) : 1 (event-driven)
w : 1
U : 1
F : 1
D : 1
'''

on_pre_ex = '''
g+=w
'''

on_pre_inh = '''
h+=w
'''

on_pre_ex_dynamic = '''
u = (1-U)*u+U
r = (r*(1-u)-1)+1
g+=w*r*u
'''

on_pre_dynamic = '''
u = (1-U)*u+U
r = (r*(1-u)-1)+1
h+=w*r*u
'''

on_pre_read = '''
g+=w
'''

# -----Neurons and Synapses setting-------
Input = NeuronGroup(n_input, neuron_in, threshold='I > 0', reset='I = 0', method='euler', refractory=0 * ms,
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

S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_read, method='euler')

S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_read, method='euler')

#-------initialization of neuron parameters----------
G_ex.v = '13.5+1.5*rand()'
G_inh.v = '13.5+1.5*rand()'
G_ex.g = '0'
G_inh.g = '0'
G_ex.h = '0'
G_inh.h = '0'

[G_ex,G_in] = base.allocate([G_ex,G_inh],3,3,15)

G_ex.run_regularly('''v = 13.5+1.5*rand()
                    g = 0
                    h = 0
                    ''',dt=duration*Dt)
G_inh.run_regularly('''v = 13.5+1.5*rand()
                    g = 0
                    h = 0
                    ''',dt=duration*Dt)

# -------initialization of network topology and synapses parameters----------
S_inE.connect(condition='j<0.3*N_post')
S_inI.connect(condition='j<0.3*N_post')
S_EE.connect(condition='i != j', p='0.3*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_EI.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_IE.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_II.connect(condition='i != j', p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_E_readout.connect(j='i')
S_I_readout.connect(j='i+n_ex')

S_inE.w = 'A_inE*randn()+A_inE'
S_inI.w = 'A_inI*randn()+A_inI'
S_EE.w = 'A_EE*randn()+A_EE'
S_IE.w = 'A_IE*randn()+A_IE'
S_EI.w = 'A_EI*randn()+A_EI'
S_II.w = 'A_II*randn()+A_II'

S_EE.delay = '1.5*ms'
S_EI.delay = '0.8*ms'
S_IE.delay = '0.8*ms'
S_II.delay = '0.8*ms'

# --------monitors setting----------
m_g_ex = StateMonitor(G_ex, (['I', 'v']), record=True)
m_g_in = StateMonitor(G_in, (['I', 'v']), record=True)
m_read = StateMonitor(G_readout, (['I', 'v']), record=True)
m_input = StateMonitor(Input, ('I'), record=True)

# ------create network-------------
net = Network(collect())


###############################################
# ------run for lms_train-------
stimulus = Time_array_train
net.store('third')
net.run(duration_train, report='text')

# ------lms_train---------------
states = base.get_states(m_read.v, duration_train, sample)
Y_train = readout.prepare_Y(label_train)
P = readout.readout(states, Y_train)
Y_train_ = readout.lms_test(states,P)
label_train_ = readout.predict_logistic(Y_train_)
score_train = readout.calculate_score(label_train, label_train_)


#####################################
# ----run for test--------
del stimulus
stimulus = Time_array_test
net.restore('third')
net.run(duration_test, report='text')

# -----lms_test-----------
states = base.get_states(m_read.v, duration_train, sample)
Y_test_ = readout.lms_test(states,P)
label_test_ = readout.predict_logistic(Y_test_)
score_test = readout.calculate_score(label_test, label_test_)

#####################################
#----------show results-----------
print('Train score: ',score_train)
print('Test score: ',score_test)


#####################################
#-----------save monitor data-------
monitor_data = {'t': m_g_ex.t/ms,
                'm_g_ex.I': m_g_ex.I,
                'm_g_ex.v': m_g_ex.v,
                'm_g_in.I': m_g_in.I,
                'm_g_in.v': m_g_in.v,
                'm_read.I': m_read.I,
                'm_read.v': m_read.v,
                'm_input.I': m_input.I}
result.result_save('monitor_temp.pkl', **monitor_data)


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
monitor = result.result_pick('monitor_temp.pkl')
play, slider, fig = result.animation(monitor['t'], monitor['m_read.v'], 50, N_test*duration)
widgets.VBox([widgets.HBox([play, slider]),fig])