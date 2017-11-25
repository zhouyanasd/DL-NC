#----------------------------------------------
# Statistics Results Using parallel computation
# Results for Spike Trains in LSM with STDP
#----------------------------------------------

from brian2 import *
from brian2tools import *
from scipy.optimize import leastsq
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
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
    Data = []
    for i in M:
        Data.append(i)
    l = len(p)
    f = p[l - 1]
    for i in range(len(Data)):
        f += p[i] * Data[i]
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


def patterns_classification(duration, patterns, neu=1, interval_l=10, interval_s=ms, percent=0.2):
    def tran_patterns(A, patterns, percent):
        trans = []
        for a in A:
            # the data is in the middle of a sequence
            for i in range(int(interval_l * percent)):
                trans.append(0)
            a_ = patterns[a]
            for i in a_:
                trans.append(int(i))
            for i in range(int(interval_l * (1 - percent))):
                trans.append(0)
        return np.asarray(trans)

    interval = interval_l + patterns.shape[1]
    if (duration / interval_s) % interval != 0:
        raise ("duration and interval+len(patterns) must be exacted division")
    n = int((duration / interval_s) / interval)
    label = np.random.randint(0, int(patterns.shape[0]), n)
    seq = tran_patterns(label, patterns, percent)
    times = where(seq == 1)[0] * interval_s
    indices = zeros(int(len(times)))
    P = SpikeGeneratorGroup(neu, indices, times)
    return P, label


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


# sample 5 times as default because the beginning is always '0'
# the states are Normalized
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
def simulate_LSM(seed):
    start_scope()
    np.random.seed(seed)
    # -----parameter and model setting-------
    patterns = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                         [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
                         [1, 0, 1, 1, 0, 1, 0, 0, 0, 1]])
    n = 4
    pre_train_duration = 1000 * ms
    duration = 1000 * ms
    duration_test = 1000 * ms
    pre_train_loop = 1
    interval_l = 40
    interval_s = ms
    threshold = 0.4
    sample = 5
    t0 = int(duration / ((interval_l + patterns.shape[1]) * interval_s))
    taupre = taupost = 2 * ms
    wmax = 0.7
    wmin = 0.3
    Apre = 0.01
    Apost = -Apre * taupre / taupost * 1.2
    equ = '''
    r : 1
    dv/dt = (I-v) / (3*ms*r) : 1 (unless refractory)
    dg/dt = (-g)/(1.5*ms*r) : 1
    dh/dt = (-h)/(1.45*ms*r) : 1
    I = tanh(g-h)*20 : 1
    '''
    equ_read = '''
    dg/dt = (-g)/(1.5*ms) : 1 
    dh/dt = (-h)/(1.45*ms) : 1
    I = tanh(g-h)*20 : 1
    '''
    on_pre = '''
    h+=w
    g+=w
    '''
    model_STDP = '''
    w : 1
    dapre/dt = -apre/taupre : 1 (clock-driven)
    dapost/dt = -apost/taupost : 1 (clock-driven)
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
    # -----neurons and synapses setting-------
    P, label = patterns_classification(duration + duration_test, patterns,
                                       interval_l=interval_l, interval_s=interval_s)
    G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                    name='neurongroup')
    G_lateral_inh = NeuronGroup(1, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                                name='neurongroup_la_inh')
    G2 = NeuronGroup(round(n / 4), equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                     name='neurongroup_1')
    G_readout = NeuronGroup(n, equ_read, method='euler')
    S = Synapses(P, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses')
    S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_1')
    S3 = Synapses(P, G_lateral_inh, 'w : 1', on_pre=on_pre, method='linear', name='synapses_2')
    S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_3')
    S5 = Synapses(G, G2, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_4')
    S6 = Synapses(G_lateral_inh, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_5')
    S_readout = Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')
    # -------network topology----------
    S.connect(j='k for k in range(n)')
    S2.connect(p=0.2)
    S3.connect()
    S4.connect(p=0.1, condition='i != j')
    S5.connect(p=0.2)
    S6.connect()
    S_readout.connect(j='i')
    S.w = '0.5+j*'+str(0.5/n)
    S2.w = '-0.5'
    S3.w = '0.8'
    S4.w = '0.3+rand()*0.4'
    S5.w = '0.3+rand()*0.4'
    S6.w = '-0.2-rand()*0.8'
    S4.delay = '3*ms'
    S.delay = '3*ms'
    G.r = '1'
    G2.r = '1'
    G_lateral_inh.r = '1'
    # ------monitors setting----------------
    m1 = StateMonitor(G_readout, ('I'), record=True, dt=ms)
    # ------create network-------------
    net = Network(collect())
    net.store('first')
    auc_test = []
    auc_train = []
    ###############################################
    for epochs in range(4):
        obj = epochs
        net.restore('first')
        patterns_pre = patterns[obj][newaxis, :]
        P_plasticity, label_plasticity = patterns_classification(pre_train_duration, patterns_pre,
                                                                 interval_l=interval_l, interval_s=interval_s)
        net.add(P_plasticity)
        S.source = P_plasticity
        S.pre.source = P_plasticity
        S._dependencies.remove(P.id)
        S.add_dependency(P_plasticity)
        S3.source = P_plasticity
        S3.pre.source = P_plasticity
        S3._dependencies.remove(P.id)
        S3.add_dependency(P_plasticity)
        net.store('first_1')
        for loop in range(pre_train_loop):
            net.run(pre_train_duration)
            net.store('second')
            net.restore('first_1')
            S4.w = net._stored_state['second']['synapses_3']['w'][0]
            S5.w = net._stored_state['second']['synapses_4']['w'][0]
        # -------change the input source----------
        net.remove(P_plasticity)
        S.source = P
        S.pre.source = P
        S._dependencies.remove(P_plasticity.id)
        S.add_dependency(P)
        S3.source = P
        S3.pre.source = P
        S3._dependencies.remove(P_plasticity.id)
        S3.add_dependency(P)
        # -------change the synapse model----------
        S4.pre.code = '''
        h+=w
        g+=w
        '''
        S4.post.code = ''
        S5.pre.code = '''
        h+=w
        g+=w
        '''
        S5.post.code = ''
        ###############################################
        # ------run for lms_train-------
        net.store('third')
        net.run(duration, report='text')
        # ------lms_train---------------
        y = label_to_obj(label[:t0], obj)
        states, _t_m = get_states(m1.I, int(interval_l + patterns.shape[1]), duration / interval_s, sample)
        Data, para = readout(states, y)
        #####################################
        # ----run for test--------
        net.restore('third')
        net.run(duration + duration_test, report='text')
        # -----lms_test-----------
        obj_t = label_to_obj(label, obj)
        states, t_m = get_states(m1.I, int(interval_l + patterns.shape[1]), (duration + duration_test) / interval_s, sample)
        y_t = lms_test(states, para)
        #####################################
        # ------calculate results----
        y_t_class, data_n = classification(threshold, y_t)
        roc_auc_train, thresholds_train = ROC(obj_t[:t0], data_n[:t0], 'ROC for train')
        print('ROC of train is %s for classification of %s' % (roc_auc_train, obj))
        roc_auc_test, thresholds_test = ROC(obj_t[t0:], data_n[t0:], 'ROC for test')
        print('ROC of test is %s for classification of %s' % (roc_auc_test, obj))
        auc_train.append(roc_auc_train)
        auc_test.append(roc_auc_test)
    return auc_train,auc_test


if __name__ == '__main__':
    tries = 2
    p = Pool(tries)
    sta_data_tri, sta_data_test =  p.map(simulate_LSM, np.arange(tries))
    print(sta_data_test,sta_data_tri)

    # ------vis of results----
    fig_tri = plt.figure(figsize=(4, 4))
    df = pd.DataFrame(np.asarray(sta_data_tri),
                      columns=['0', '1', '2', '3'])
    df.boxplot()
    plt.title('Classification Condition of train')

    fig_test = plt.figure(figsize=(4, 4))
    df = pd.DataFrame(np.asarray(sta_data_test),
                      columns=['0', '1', '2', '3'])
    df.boxplot()
    plt.title('Classification Condition of test')
    show()
