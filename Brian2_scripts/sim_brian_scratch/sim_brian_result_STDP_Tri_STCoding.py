#----------------------------------------------
# Statistics Results Using parallel computation
# Results for ST-Coding Tri-function in LSM with STDP
#----------------------------------------------

from brian2 import *
from brian2tools import *
from scipy.optimize import leastsq
import scipy as sp
import pandas as pd
from multiprocessing import Queue,Pool
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

prefs.codegen.target = "numpy"  # it is faster than use default "cython"

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


def ROC(y, scores, pos_label=1):
    def normalization_min_max(arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
            arr_n[i] = x
        return arr_n

    scores_n = normalization_min_max(scores)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores_n, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc, thresholds


def get_states(input, interval, duration, sample=5):
    n = int(duration / interval)
    step = int(interval / sample)
    temp = []
    for i in range(n):
        sum = np.sum(input[:, i * interval:(i + 1) * interval:step], axis=1)
        temp.append(sum)
    return MinMaxScaler().fit_transform(np.asarray(temp).T)


###############################################
def simulate_LSM(seed):
    start_scope()
    np.random.seed(seed)

    # -----parameter and model setting-------
    n = 20
    pre_train_duration = 3000 * ms
    duration = 3000 * ms
    duration_test = 3000 * ms
    pre_train_loop = 0
    interval_s = defaultclock.dt
    threshold = 0.5
    pattern_duration = 250
    pattern_interval = 150
    sample = 10

    t0 = int(duration / (pattern_duration * interval_s))
    t1 = int((duration + duration_test) / (pattern_duration * interval_s))

    taupre = taupost = 2 * ms
    wmax = 0.6
    wmin = 0.2
    Apre = 0.003
    Apost = -Apre * taupre / taupost * 1.2

    equ_in = '''
    dv/dt = (I-v) / (1.5*ms) : 1 (unless refractory)
    I = stimulus(t)*0.7: 1
    '''

    equ = '''
    r : 1
    I_0 : 1 
    dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
    dg/dt = (-g)/(1.5*ms*r) : 1
    dh/dt = (-h)/(1.45*ms*r) : 1
    I = tanh(g-h)*20 +I_0: 1
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
    data, label = Tri_function(duration + duration_test, pattern_duration=pattern_duration,
                               pattern_interval=pattern_interval)

    Time_array = TimedArray(data, dt=defaultclock.dt)

    Input = NeuronGroup(1, equ_in, threshold='v > 0.20', reset='v = 0', method='euler', refractory=0.1 * ms,
                        name='neurongroup_input')

    G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory= 1 * ms,
                    name='neurongroup')

    G2 = NeuronGroup(int(n / 4), equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory= 1 * ms,
                     name='neurongroup_1')

    G_lateral_inh = NeuronGroup(1, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory= 1 * ms,
                                name='neurongroup_la_inh')

    G_readout = NeuronGroup(n, equ_read, method='euler', name='neurongroup_read')

    S = Synapses(Input, G, 'w : 1', on_pre = on_pre ,method='linear', name='synapses')

    S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_1')

    S3 = Synapses(Input, G_lateral_inh, 'w : 1', on_pre = on_pre , method='linear', name='synapses_2')

    S5 = Synapses(G, G2, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_4')

    S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_3')

    S6 = Synapses(G_lateral_inh, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_5')

    S_readout = Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

    # -------network topology----------
    S.connect(j='k for k in range(int(n*1))')
    S2.connect(p=0.2)
    S3.connect()
    S4.connect(p=0.1, condition='i != j')
    S5.connect(p=0.2)
    S6.connect(j='k for k in range(int(n*1))')
    S_readout.connect(j='i')

    S.w = '0.6+j*'+str(0.4/n)
    S2.w = '-0.5'
    S3.w = '0.95'
    S4.w = '0.2+rand()*0.4'
    S5.w = '0.2+rand()*0.4'
    S6.w = '-0.4-rand()*0.6'

    S.delay = '3*ms'
    S4.delay = '3*ms'

    G.r = '1'
    G2.r = '1'
    G_lateral_inh.r = '1'

    # ------monitor----------------
    m_read = StateMonitor(G_readout, ('I'), record=True)

    # ------create network-------------
    net = Network(collect())
    net.store('first')
    auc_test = []
    auc_train = []
    ###############################################
    # ------pre_train------------------
    for epochs in range(3):
        obj = epochs
        net.restore('first')
        S5.pre.code = S4.pre.code = on_pre_STDP
        S5.post.code = S4.post.code = on_post_STDP
        data_pre, label_pre = Tri_function(pre_train_duration, pattern_duration=pattern_duration,
                                           pattern_interval=pattern_interval, obj=obj)
        Time_array_pre = TimedArray(data_pre, dt=defaultclock.dt)
        stimulus = Time_array_pre

        for loop in range(pre_train_loop):
            net.run(pre_train_duration)
            net.store('second')
            net.restore('first')
            S4.w = net._stored_state['second']['synapses_3']['w'][0]
            S5.w = net._stored_state['second']['synapses_4']['w'][0]

        # -------change the synapse model----------
        stimulus = Time_array

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
        states = get_states(m_read.I, pattern_duration, duration / interval_s, sample)
        Data, para = readout(states, y)

        #####################################
        # ----run for test--------
        net.restore('third')
        net.run(duration + duration_test)

        # -----lms_test-----------
        y = label_to_obj(label, obj)
        states = get_states(m_read.I, pattern_duration, (duration + duration_test) / interval_s, sample)
        y_t = lms_test(states, para)

        y_t_class, data_n = classification(threshold, y_t)
        roc_auc_train, thresholds_train = ROC(y[:t0], data_n[:t0])
        print('ROC of train is %s for classification of %s' % (roc_auc_train, obj))
        roc_auc_test, thresholds_test = ROC(y[t0:], data_n[t0:])
        print('ROC of test is %s for classification of %s' % (roc_auc_test, obj))
        auc_train.append(roc_auc_train)
        auc_test.append(roc_auc_test)
    return [auc_train, auc_test]

if __name__ == '__main__':
    tries = 10
    p = Pool(tries)
    result = p.map(simulate_LSM, np.arange(tries))
    sta_data_tri, sta_data_test = [x[0] for x in result], [x[1] for x in result]
    print(sta_data_test,sta_data_tri)

    # ------vis of results----
    fig_tri = plt.figure(figsize=(4, 4))
    df = pd.DataFrame(np.asarray(sta_data_tri),
                      columns=['0', '1', '2'])
    df.boxplot()
    plt.title('Classification Condition of train')

    fig_test = plt.figure(figsize=(4, 4))
    df = pd.DataFrame(np.asarray(sta_data_test),
                      columns=['0', '1', '2'])
    df.boxplot()
    plt.title('Classification Condition of test')
    show()