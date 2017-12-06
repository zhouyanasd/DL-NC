#----------------------------------------------
# Statistics Results Using parallel computation
# Results for Tri-function in LSM with STDP
#----------------------------------------------

from brian2 import *
from brian2tools import *
from scipy.optimize import leastsq
import scipy as sp
import pandas as pd
from multiprocessing import Queue,Pool

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


def Tri_function(duration, obj=-1):
    rng = np.random
    TIME_SCALE = defaultclock.dt
    in_number = int(duration / TIME_SCALE)

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

    data = []
    cla = []
    cons = rng.randint(1, 101)
    fun, c = chose_fun()

    for t in range(in_number):
        if change_fun(0.7) and t % 100 == 0:
            cons = rng.randint(1, 101)
            fun, c = chose_fun()
            try:
                data_t = fun(data[t - 1], cons, t)
                data.append(data_t)
                cla.append(c)
            except IndexError:
                data_t = fun(rng.randint(1, 101) / 100, cons, t)
                data.append(data_t)
                cla.append(c)
        else:
            try:
                data_t = fun(data[t - 1], cons, t)
                data.append(data_t)
                cla.append(c)
            except IndexError:
                data_t = fun(rng.randint(1, 101) / 100, cons, t)
                data.append(data_t)
                cla.append(c)
    cla = np.asarray(cla)
    data = np.asarray(data)
    return data, cla


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
    return roc_auc, thresholds


###############################################
def simulate_LSM(seed):
    start_scope()
    np.random.seed(seed)

    # -----parameter and model setting-------
    n = 20
    pre_train_duration = 200 * ms
    duration = 200 * ms
    duration_test = 200 * ms
    pre_train_loop = 0
    interval_s = defaultclock.dt
    threshold = 0.5

    t0 = int(duration / interval_s)
    t1 = int((duration + duration_test) / interval_s)

    taupre = taupost = 0.5 * ms
    wmax = 0.5
    wmin = 0.1
    Apre = 0.003
    Apost = -Apre * taupre / taupost * 1.2

    equ_in = '''
    I = stimulus(t) : 1
    '''

    equ = '''
    r : 1
    I_0 : 1 
    dv/dt = (I-v) / (0.3*ms) : 1 (unless refractory)
    dg/dt = (-g)/(0.15*ms*r) : 1
    dh/dt = (-h)/(0.145*ms*r) : 1
    I = tanh(g-h)*20 +I_0: 1
    '''

    equ_read = '''
    dg/dt = (-g)/(0.6*ms) : 1 
    dh/dt = (-h)/(0.58*ms) : 1
    I = tanh(g-h)*20 : 1
    '''

    model_STDP = '''
    w : 1
    dapre/dt = -apre/taupre : 1 (clock-driven)
    dapost/dt = -apost/taupost : 1 (clock-driven)
    '''

    on_pre_input = '''
    I_0 = 0 
    I_0 += w * I_pre
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
    data, label = Tri_function(duration + duration_test)
    stimulus = TimedArray(data, dt=defaultclock.dt)

    Input = NeuronGroup(1, equ_in, method='linear', events={'input':'True'}, name = 'neurongroup_input')

    G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=0.1 * ms,
                    name='neurongroup')

    G2 = NeuronGroup(int(n / 4), equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=0.1 * ms,
                     name='neurongroup_1')

    G_lateral_inh = NeuronGroup(1, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=0.1 * ms,
                                name='neurongroup_la_inh')

    G_readout = NeuronGroup(n, equ_read, method='euler', name='neurongroup_read')

    S = Synapses(Input, G, 'w : 1', on_pre = on_pre_input ,method='linear', on_event={'pre':'input'}, name='synapses')

    S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_1')

    S3 = Synapses(Input, G_lateral_inh, 'w : 1', on_pre = on_pre_input ,method='linear', on_event={'pre':'input'},
                  name='synapses_2')

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

    S.w = '0.5+j*'+str(0.4/n)
    S2.w = '-0.4'
    S3.w = '0.5'
    S4.w = '0.1+rand()*0.4'
    S5.w = '0.1+rand()*0.4'
    S6.w = '-0.2-rand()*0.8'

    S.delay = '0.3*ms'
    S4.delay = '0.3*ms'

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
        data_pre, label_pre = Tri_function(pre_train_duration, obj=obj)
        stimulus.values = data_pre

        for loop in range(pre_train_loop):
            net.run(pre_train_duration)
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
        Data, para = readout(m_read.I, y)

        #####################################
        # ----run for test--------
        net.restore('third')
        net.run(duration + duration_test)

        # -----lms_test-----------
        y = label_to_obj(label, obj)
        y_t = lms_test(m_read.I, para)

        y_t_class, data_n = classification(threshold, y_t)
        roc_auc_train, thresholds_train = ROC(y[:t0], data_n[:t0], 'ROC for train of %s' % obj)
        print('ROC of train is %s for classification of %s' % (roc_auc_train, obj))
        roc_auc_test, thresholds_test = ROC(y[t0:], data_n[t0:], 'ROC for test of %s' % obj)
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