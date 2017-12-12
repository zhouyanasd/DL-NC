# ----------------------------------------
# LSM for Tri-funcion test
# multiple pre-train STDP and the distribution is different for different patterns
# add Input layer as input and the encoding is transformed into spike trains
# simulation 7--analysis 3
# ----------------------------------------

from brian2 import *
from brian2tools import *
from scipy.optimize import leastsq
import scipy as sp
from sklearn.preprocessing import MinMaxScaler

prefs.codegen.target = "numpy"  # it is faster than use default "cython"
start_scope()
np.random.seed(103)


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


def lms_test(M, p):
    l = len(p)
    f = p[l - 1]
    for i in range(len(M)):
        f += p[i] * M[i]
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


def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))


def Tri_function(duration, pattern_duration = 100, pattern_interval = 50, obj=-1):
    rng = np.random
    TIME_SCALE = defaultclock.dt
    in_number = int(duration / TIME_SCALE)
    data = []
    cla = []

    def sin_fun(l, c, t):
        return (np.sin(c * t * TIME_SCALE / us) + 1) / 2

    def tent_map(l, c, t):
        temp = l
        if (temp < 0.5 and temp > 0):
            temp = (c / 101 + 1) * temp
            return temp
        elif (temp >= 0.5 and temp < 1):
            temp = (c / 101 + 1) * (1 - temp)
            return temp
        else:
            return 0.5

    def constant(l, c, t):
        return c / 100

    def chose_fun():
        if obj == -1:
            c = rng.randint(0, 3)
        else:
            c = obj
        if c == 0:
            return sin_fun, c
        elif c == 1:
            return tent_map, c
        elif c == 2:
            return constant, c

    def change_fun(rate):
        fun = rng.randint(1, 101)
        if fun > 100 * rate:
            return False
        else:
            return True

    for t in range(in_number):
        t_temp = t % pattern_duration
        if t_temp == 0:
            cons = rng.randint(1, 101)
            fun, c = chose_fun()
            cla.append(c)

        if t_temp < pattern_duration - pattern_interval:
            try:
                data_t = fun(data[t - 1], cons, t)
                data.append(data_t)
            except IndexError:
                data_t = fun(rng.randint(1, 101) / 100, cons, t)
                data.append(data_t)
        elif t_temp >= pattern_duration - pattern_interval:
            data.append(0)
        else :
            data.append(0)
    return np.asarray(data), np.asarray(cla)


def label_to_obj(label, obj):
    temp = []
    for a in label:
        if a == obj:
            temp.append(1)
        else:
            temp.append(0)
    return np.asarray(temp)


def classification(thea, data):
    def normalization_min_max(arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
            arr_n[i] = x
        return arr_n

    data_n = normalization_min_max(data)
    data_class = []
    for a in data_n:
        if a >= thea:
            b = 1
        else:
            b = 0
        data_class.append(b)
    return np.asarray(data_class), data_n


def ROC(y, scores, fig_title='ROC', pos_label=1):
    def normalization_min_max(arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
            arr_n[i] = x
        return arr_n

    scores_n = normalization_min_max(scores)
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y, scores_n, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)

    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(fig_title)
    plt.legend(loc="lower right")
    return fig, roc_auc, thresholds


def get_states(input, interval, duration, sample=5):
    n = int(duration / interval)
    t = np.arange(n) * interval
    step = int(interval / sample)
    temp = []
    for i in range(n):
        sum = np.sum(input[:, i * interval:(i + 1) * interval:step], axis=1)
        temp.append(sum)
    return MinMaxScaler().fit_transform(np.asarray(temp).T), t


###############################################
# -----parameter and model setting-------
obj = 1
n = 20
pre_train_duration = 1000 * ms
duration = 1000 * ms
duration_test = 1000 * ms
pre_train_loop = 0
interval_s = defaultclock.dt
threshold = 0.5
pattern_duration = 250
pattern_interval = 150
sample = 10


t0 = int(duration / (pattern_duration*interval_s))
t1 = int((duration + duration_test) / (pattern_duration*interval_s))

taupre = taupost = 2 * ms
wmax = 0.7
wmin = 0.3
Apre = 0.003
Apost = -Apre * taupre / taupost * 1.2

equ_in = '''
dv/dt = (I-v) / (1.5*ms) : 1 (unless refractory)
I = stimulus(t)*0.7: 1
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
data_pre, label_pre = Tri_function(pre_train_duration, pattern_duration = pattern_duration,
                                   pattern_interval = pattern_interval, obj=obj)
data, label = Tri_function(duration + duration_test, pattern_duration = pattern_duration,
                           pattern_interval = pattern_interval)
stimulus = TimedArray(data, dt=defaultclock.dt)

Input = NeuronGroup(1, equ_in, threshold='v > 0.20', reset='v = 0', method='euler', refractory=0.1 * ms,
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
S.connect(j='k for k in range(int(n*1))')
S2.connect(p=1)
S3.connect()
S4.connect(p=1, condition='i != j')
S5.connect(p=1)
S6.connect(j='k for k in range(int(n*1))')
S_readout.connect(j='i')

S.w = '0.6+j*'+str(0.4/n)
S2.w = '-1'
S3.w = '1'
S4.w = 'rand()'
S5.w = 'rand()'
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
# print('S4.w = %s' % S4.w)
###############################################
# ------pre_train------------------
stimulus.values = data_pre
for loop in range(pre_train_loop):
    net.run(pre_train_duration)

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
stimulus.values = data

S5.pre.code = S4.pre.code = '''
h+=w
g+=w
'''
S5.post.code = S4.post.code = ''


###############################################
# ------run for lms_train-------
net.store('third')
net.run(duration)

# ------lms_train---------------
y = label_to_obj(label[:t0], obj)
states, _t_m = get_states(m_read.I, pattern_duration, duration / interval_s, sample)
Data, para = readout(states, y)

#####################################
# ----run for test--------
net.restore('third')
net.run(duration + duration_test)

# -----lms_test-----------
y = label_to_obj(label, obj)
states, t_m = get_states(m_read.I, pattern_duration, (duration + duration_test) / interval_s, sample)
y_t = lms_test(states, para)

y_t_class, data_n = classification(threshold, y_t)
fig_roc_train, roc_auc_train, thresholds_train = ROC(y[:t0], data_n[:t0], 'ROC for train of %s' % obj)
print('ROC of train is %s for classification of %s' % (roc_auc_train, obj))
fig_roc_test, roc_auc_test, thresholds_test = ROC(y[t0:], data_n[t0:], 'ROC for test of %s' % obj)
print('ROC of test is %s for classification of %s' % (roc_auc_test, obj))

#####################################
# ------vis of results----
fig0 = plt.figure(figsize=(20, 4))
subplot(211)
plot(data_pre, 'r')
subplot(212)
plot(data, 'r')

fig1 = plt.figure(figsize=(20, 8))
subplot(111)
plt.scatter(t_m, y_t_class, s=2, color="red", marker='o', alpha=0.6)
plt.scatter(t_m, y, s=3, color="blue", marker='*', alpha=0.4)
plt.scatter(t_m, data_n, s=2, color="green")
axhline(threshold, ls='--', c='r', lw=1)
plt.title('Classification Condition of threshold = %s' % threshold)

fig3 = plt.figure(figsize=(20, 8))
subplot(411)
plt.plot(m_g.t / ms, m_g.v.T, label='v')
legend(labels=[('V_%s' % k) for k in range(n)], loc='upper right')
subplot(412)
plt.plot(m_g.t / ms, m_g.I.T, label='I')
legend(labels=[('I_%s' % k) for k in range(n)], loc='upper right')
subplot(413)
plt.plot(m_in.t / ms, m_in.v.T, label='v')
legend(labels=[('V_%s' % k) for k in range(n)], loc='upper right')
subplot(414)
plt.plot(m_in.t / ms, m_in.I.T, label='I')
legend(labels=[('I_%s' % k) for k in range(n)], loc='upper right')

fig4 = plt.figure(figsize=(20, 8))
subplot(411)
plt.plot(m_g2.t / ms, m_g2.v.T, label='v')
legend(labels=[('V_%s' % k) for k in range(n)], loc='upper right')
subplot(412)
plt.plot(m_g2.t / ms, m_g2.I.T, label='I')
legend(labels=[('I_%s' % k) for k in range(n)], loc='upper right')
subplot(413)
plt.plot(m_inh.t / ms, m_inh.v.T, label='v')
legend(labels=[('V_%s' % k) for k in range(n)], loc='upper right')
subplot(414)
plt.plot(m_inh.t / ms, m_inh.I.T, label='I')
legend(labels=[('V_%s' % k) for k in range(n)], loc='upper right')


fig5 = plt.figure(figsize=(20, 4))
plt.plot(m_read.t / ms, m_read.I.T, label='I')
legend(labels=[('I_%s' % k) for k in range(n)], loc='upper right')

fig6 =plt.figure(figsize=(4,4))
brian_plot(S4.w)
show()
