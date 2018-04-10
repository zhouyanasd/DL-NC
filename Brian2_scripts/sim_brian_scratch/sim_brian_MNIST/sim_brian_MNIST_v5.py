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
def softmax(z):
    return np.array([(np.exp(i) / np.sum(np.exp(i))) for i in z])

def train(X, Y, P):
    a = 0.0001
    max_iteration = 10000
    time = 0
    while time < max_iteration:
        time += 1
        P = P + X.T.dot(Y - softmax(X.dot(P))) * a
    return P

def lms_test(Data, p):
    one = np.ones((Data.shape[1], 1)) #bis
    X = np.hstack((Data.T, one))
    return X.dot(p)

def readout(M, Y):
    one = np.ones((M.shape[1], 1))
    X = np.hstack((M.T, one))
    P = np.random.rand(X.shape[1],Y.T.shape[1])
    para = train(X, Y.T, P)
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


def get_states(input, interval, duration, sample):
    n = int(duration / interval)
    step = int(interval / sample / defaultclock.dt)
    interval_ = int(interval / defaultclock.dt)
    temp = []
    for i in range(n):
        sum = np.sum(input[:, i * interval_: (i + 1) * interval_][:,::-step], axis=1)
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
        im.append(np.reshape(temp, (1, 784)))
        image_index += struct.calcsize('>784B')

    label_index = 0
    label_index += struct.calcsize('>II')
    label = np.asarray(struct.unpack_from('>' + str(n) + 'B', buf2, label_index))

    f = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    df = pd.DataFrame({'value': pd.Series(im).apply(f), 'label': pd.Series(label)})
    return df


def encoding_latency_MNIST(analog_data, n, duration, min = 0, max = np.pi, *args):
    def encoding_cos(x, n, A):
        encoding = []
        for i in range(int(n)):
            trans_cos = np.round(0.5*A*(np.cos(x+np.pi*(i/n))+1)).clip(0,A-1)
            coding = [([0] * trans_cos.shape[1]) for i in range(A*trans_cos.shape[0])]
            index_0 = 0
            for p in trans_cos:
                index_1 = 0
                for q in p:
                    coding[int(q)][index_1+A*index_0] = 1
                    index_1 += 1
                index_0 += 1
            encoding.extend(coding)
        return np.asarray(encoding)
    f = lambda x: (max-min)*(x - np.min(x)) / (np.max(x) - np.min(x))
    return analog_data.apply(f).apply(encoding_cos, n = n, A = duration)


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
    for value in encoding_latency_MNIST(data_frame_obj['value'][:n], kwargs['coding_n'], int(0.1*duration)):
        for data in value:
            data_frame_s.append(list(data))
        interval = duration - value.shape[0]
        if interval > 0.3*duration:
            data_frame_s.extend([[0] * 784] * interval)
        else:
            raise Exception('duration is too short')
    data_frame_s = np.asarray(data_frame_s)
    label = data_frame_obj['label'][:n]
    return data_frame_s, label


def result_save(path, *arg, **kwarg):
    fw = open(path, 'wb')
    pickle.dump(kwarg, fw)
    fw.close()


def result_pick(path):
    fr = open(path, 'rb')
    data = pickle.load(fr)
    fr.close()
    return data


def animation(t, v, interval, duration, a_step=10, a_interval=100, a_duration = 10):

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


def allocate(G, X, Y, Z):
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


def w_norm2(n_post, Synapsis):
    for i in range(n_post):
        a = Synapsis.w[np.where(Synapsis._synaptic_post == i)[0]]
        Synapsis.w[np.where(Synapsis._synaptic_post == i)[0]] = a/np.linalg.norm(a)


# -----parameter setting-------
duration = 1000
N_train = 100
N_test = 100
Dt = defaultclock.dt
sample = 2
pre_train_loop = 0

n = 108
R = 2
A_EE = 30
A_EI = 60
A_IE = -19
A_II = -19
A_inE = 18
A_inI = 9

U_EE = 0.5
U_EI = 0.05
U_IE = 0.25
U_II = 0.32

D_EE = 1.1
D_EI = 0.125
D_IE = 0.7
D_II = 0.144

F_EE = 0.05
F_EI = 1.2
F_IE = 0.02
F_II = 0.06


df_train = load_Data_MNIST(60000, '../../../Data/MNIST_data/train-images.idx3-ubyte',
                               '../../../Data/MNIST_data/train-labels.idx1-ubyte')
df_test = load_Data_MNIST(10000, '../../../Data/MNIST_data/t10k-images.idx3-ubyte',
                               '../../../Data/MNIST_data/t10k-labels.idx1-ubyte')

data_train_s, label_train = get_series_data(N_train, df_train, duration, False, coding_n = 3)
data_test_s, label_test = get_series_data(N_test, df_test, duration, False, coding_n = 3)

duration_train = len(data_train_s) * Dt
duration_test = len(data_test_s) * Dt

equ_in = '''
I = stimulus(t,i) : 1
'''

equ = '''
dv/dt = (I-v) / (30*ms) : 1 (unless refractory)
dg/dt = (-g)/(3*ms) : 1
dh/dt = (-h)/(6*ms) : 1
I = (g+h)+13.5: 1
x : 1
y : 1
z : 1
'''

equ_read = '''
dv/dt = (I-v) / (30*ms) : 1
dg/dt = (-g)/(3*ms) : 1 
dh/dt = (-h)/(6*ms) : 1
I = (g+h): 1
'''

on_pre_ex = '''
u = (1-U)*u+U
r = (r*(1-u)-1)+1
g+=w*r*u
'''

on_pre_inh = '''
u = (1-U)*u+U
r = (r*(1-u)-1)+1
h+=w*r*u
'''

on_pre_read = '''
g+=w
'''

dynamic_synapse = '''
du/dt = (-u)/(F*second) : 1 (event-driven)
dr/dt = (-r)/(D*second) : 1 (event-driven)
w : 1
U : 1
F : 1
D : 1
'''

# -----simulation setting-------
Time_array_train = TimedArray(data_train_s, dt=Dt)

Time_array_test = TimedArray(data_test_s, dt=Dt)

Input = NeuronGroup(784, equ_in, threshold='I > 0', reset='I = 0', method='euler', refractory=0 * ms,
                    name = 'neurongroup_input')

G_ex = NeuronGroup(n, equ, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                name='neurongroup_ex')

G_inh = NeuronGroup(int(n/4), equ, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                name='neurongroup_in')

G_readout = NeuronGroup(int(n*5/4), equ_read, method='euler', name='neurongroup_read')

S_inE = Synapses(Input, G_ex, dynamic_synapse, on_pre = on_pre_ex ,method='euler', name='synapses_inE')

S_inI = Synapses(Input, G_inh, dynamic_synapse, on_pre = on_pre_ex ,method='euler', name='synapses_inI')

S_EE = Synapses(G_ex, G_ex, dynamic_synapse, on_pre = on_pre_ex ,method='euler', name='synapses_EE')

S_EI = Synapses(G_ex, G_inh, dynamic_synapse, on_pre = on_pre_ex ,method='euler', name='synapses_EI')

S_IE = Synapses(G_inh, G_ex, dynamic_synapse, on_pre = on_pre_inh ,method='euler', name='synapses_IE')

S_II = Synapses(G_inh, G_inh, dynamic_synapse, on_pre = on_pre_inh ,method='euler', name='synapses_II')

S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_read, method='euler')

S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_read, method='euler')

# -------network topology----------
G_ex.v = '13.5+1.5*rand()'
G_inh.v = '13.5+1.5*rand()'
[G_ex,G_in] = allocate([G_ex,G_inh],3,3,15)
G_ex.run_regularly('''v = 13.5+1.5*rand()
                    g = 0
                    h = 0
                    I = 13.5''',dt=duration*Dt)
G_inh.run_regularly('''v = 13.5+1.5*rand()
                    g = 0
                    h = 0
                    I = 13.5''',dt=duration*Dt)

S_inE.connect(condition='j<0.3*N_post')
S_inI.connect(condition='j<0.3*N_post')
S_EE.connect(condition='i != j', p='0.3*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_EI.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_IE.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_II.connect(condition='i != j', p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
S_E_readout.connect(j='i')
S_I_readout.connect(j='i+n')

S_inE.w = 'A_inE*randn()+A_inE'
w_norm2(n, S_inE)
S_inI.w = 'A_inI*randn()+A_inI'
w_norm2(int(n/4), S_inI)

S_EE.w = 'A_EE*randn()+A_EE'
S_IE.w = 'A_IE*randn()+A_IE'
S_EI.w = 'A_EI*randn()+A_EI'
S_II.w = 'A_II*randn()+A_II'

S_inE.U = 'U_EE*randn()+U_EE'
S_inI.U = 'U_EI*randn()+U_EI'
S_EE.U = 'U_EE*randn()+U_EE'
S_IE.U = 'U_IE*randn()+U_IE'
S_EI.U = 'U_EI*randn()+U_EI'
S_II.U = 'U_II*randn()+U_II'

S_inE.F = 'F_EE*randn()+F_EE'
S_inI.F = 'F_EI*randn()+F_EI'
S_EE.F = 'F_EE*randn()+F_EE'
S_IE.F = 'F_IE*randn()+F_IE'
S_EI.F = 'F_EI*randn()+F_EI'
S_II.F = 'F_II*randn()+F_II'

S_inE.D = 'D_EE*randn()+D_EE'
S_inI.D = 'D_EI*randn()+D_EI'
S_EE.D = 'D_EE*randn()+D_EE'
S_IE.D = 'D_IE*randn()+D_IE'
S_EI.D = 'D_EI*randn()+D_EI'
S_II.D = 'D_II*randn()+D_II'

S_inE.u = '0'
S_inI.u = '0'
S_EE.u = '0'
S_IE.u = '0'
S_EI.u = '0'
S_II.u = '0'

S_inE.r = '0'
S_inI.r = '0'
S_EE.r = '0'
S_IE.r = '0'
S_EI.r = '0'
S_II.r = '0'

S_EE.delay = '1.5*ms'
S_EI.delay = '0.8*ms'
S_IE.delay = '0.8*ms'
S_II.delay = '0.8*ms'

# ------monitor----------------
m_g_ex = StateMonitor(G_ex, (['I', 'v']), record=True)
m_g_in = StateMonitor(G_in, (['I', 'v']), record=True)
m_read = StateMonitor(G_readout, ('v'), record=True)
m_input = StateMonitor(Input, ('I'), record=True)

# ------create network-------------
net = Network(collect())

###############################################
# ------run for lms_train-------
stimulus = Time_array_train
net.store('third')
net.run(duration_train, report='text')

# ------lms_train---------------
Y_train = one_versus_the_rest(label_train, selected=np.arange(10))
states = get_states(m_read.v, duration*Dt , duration_train, sample)
P = readout(states, Y_train)
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
states = get_states(m_read.v, duration*Dt , duration_test, sample)
Y_test_ = lms_test(states,P)
label_test_ = trans_max_to_label(Y_test_)
score_test = accuracy_score(label_test, label_test_)

#####################################
#----------show results-----------
print('Train score: ',score_train)
print('Test score: ',score_test)

#####################################
#-----------save monitor data-------
monitor_data = {'t':m_g_ex.t/ms,
                'm_g_ex.I':m_g_ex.I,
                'm_g_ex.v':m_g_ex.v,
                'm_g_in.I': m_g_in.I,
                'm_g_in.v': m_g_in.v,
                'm_read.v':m_read.v,
                'm_input.I':m_input.I}
result_save('monitor_temp.pkl', **monitor_data)


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
monitor = result_pick('monitor_temp.pkl')
play, slider, fig = animation(monitor['t'], monitor['m_read.v'], 50, N_test*duration)
widgets.VBox([widgets.HBox([play, slider]),fig])