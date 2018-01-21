#----------------------------------------------
# Statistics Results Using parallel computation
# Results for ST-Coding JV in LSM with STDP
#----------------------------------------------

from brian2 import *
from brian2tools import *
from scipy.optimize import leastsq
import scipy as sp
import pandas as pd
from multiprocessing import Queue,Pool
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

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


def lms_test(Data, p):
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


def ROC(y, scores, pos_label=1):
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


def get_states(input, interval, duration, sample):
    n = int(duration / interval)
    t = np.arange(n) * interval / defaultclock.dt
    step = int(interval / sample / defaultclock.dt)
    interval_ = int(interval / defaultclock.dt)
    temp = []
    for i in range(n):
        sum = np.sum(input[:, i * interval_: (i + 1) * interval_: step], axis=1)
        temp.append(sum)
    return MinMaxScaler().fit_transform(np.asarray(temp).T), t


def load_Data_JV(t, path_value="../../Data/jv/train.txt", path_label="../../Data/jv/size.txt"):
    if t == "train":
        label = np.loadtxt(path_label, delimiter=None).astype(int)[1]
    elif t == "test":
        label = np.loadtxt(path_label, delimiter=None).astype(int)[0]
    else:
        raise TypeError("t must be 'train' or 'test'")
    data = np.loadtxt(path_value, delimiter=None)
    data = MinMaxScaler().fit_transform(data)
    s = open(path_value, 'r')
    i = -1
    size_d = []
    while True:
        lines = s.readline()
        i += 1
        if not lines:
            break
        if lines == '\n':  # "\n" needed to be added at the end of the file
            i -= 1
            size_d.append(i)
            continue
    size_d = np.asarray(size_d) + 1
    size_d = np.concatenate(([0], size_d))
    data_list = [data[size_d[i]:size_d[i + 1]] for i in range(len(size_d) - 1)]
    label_list = []
    j = 0
    for n in label:
        label_list.extend([j] * n)
        j += 1
    data_frame = pd.DataFrame({'value': pd.Series(data_list), 'label': pd.Series(label_list)})
    return data_frame


def get_series_data(data_frame, duration, is_order = True, *args, **kwargs):
    try:
        obj = kwargs['obj']
    except KeyError:
        obj = np.arange(9)
    if not is_order:
        data_frame_obj = data_frame[data_frame['label'].isin(obj)].sample(frac=1).reset_index(drop=True)
    else:
        data_frame_obj = data_frame[data_frame['label'].isin(obj)]
    data_frame_s = []
    for value in data_frame_obj['value']:
        for data in value:
            data_frame_s.append(list(data))
        interval = duration - value.shape[0]
        if interval >30 :
            data_frame_s.extend([[0]*12]*interval)
        else:
            raise Exception('duration is too short')
    data_frame_s = np.asarray(data_frame_s)
    label = data_frame_obj['label']
    return data_frame_s, label

###############################################
def simulate_LSM(seed):
    start_scope()
    np.random.seed(seed)

    # -----parameter setting-------
    duration = 200
    Dt = defaultclock.dt * 2
    n = 20
    pre_train_loop = 0
    sample = 10

    data_train = load_Data_JV(t='train', path_value="../../Data/jv/train.txt")
    data_test = load_Data_JV(t='test', path_value="../../Data/jv/test.txt")

    data_train_s, label_train = get_series_data(data_train, duration)
    data_test_s, label_test = get_series_data(data_test, duration)

    duration_train = len(data_train_s) * Dt
    duration_test = len(data_test_s) * Dt

    equ_in = '''
    dv/dt = (I-v) / (1.5*ms) : 1 (unless refractory)
    I = stimulus(t,i) : 1
    '''

    equ = '''
    r : 1
    dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
    dg/dt = (-g)/(1.5*ms*r) : 1
    dh/dt = (-h)/(1.45*ms*r) : 1
    I = tanh(g-h)*5: 1
    '''

    equ_read = '''
    dg/dt = (-g)/(1.5*ms) : 1 
    dh/dt = (-h)/(1.45*ms) : 1
    I = tanh(g-h)*5 : 1
    '''

    model_STDP = '''
    w : 1
    wmax : 1
    wmin : 1
    Apre : 1
    Apost = -Apre * taupre / taupost * 1.2 : 1
    taupre : second
    taupost : second
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
    Time_array_train = TimedArray(data_train_s, dt=Dt)

    Time_array_test = TimedArray(data_test_s, dt=Dt)

    Input = NeuronGroup(12, equ_in, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                        name='neurongroup_input')

    G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                    name='neurongroup')

    G2 = NeuronGroup(int(n / 4), equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                     name='neurongroup_1')

    G_lateral_inh = NeuronGroup(1, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                                name='neurongroup_la_inh')

    G_readout = NeuronGroup(n, equ_read, method='euler', name='neurongroup_read')

    S = Synapses(Input, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses')

    S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_1')

    S3 = Synapses(Input, G_lateral_inh, 'w : 1', on_pre=on_pre, method='linear', name='synapses_2')

    S5 = Synapses(G, G2, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_4')

    S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_3')

    S6 = Synapses(G_lateral_inh, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_5')

    S_readout = Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

    # -------network topology----------
    S.connect(j='k for k in range(int(n))')
    S2.connect(p=0.2)
    S3.connect()
    S4.connect(p=0.1, condition='i != j')
    S5.connect(p=0.2)
    S6.connect(j='k for k in range(int(n))')
    S_readout.connect(j='i')

    S4.wmax = '0.5+rand()*0.4'
    S5.wmax = '0.5+rand()*0.4'
    S4.wmin = '0.2+rand()*0.2'
    S5.wmin = '0.2+rand()*0.2'
    S4.Apre = S5.Apre = '0.01'
    S4.taupre = S4.taupost = '1*ms+rand()*4*ms'
    S5.taupre = S5.taupost = '1*ms+rand()*4*ms'

    S.w = '0.2+j*' + str(0.6 / (n))
    S2.w = '-0.2'
    S3.w = '0.8'
    S4.w = 'wmin+rand()*(wmax-wmin)'
    S5.w = 'wmin+rand()*(wmax-wmin)'
    S6.w = '-rand()'

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
    for epochs in range(12):
        obj = epochs
        net.restore('first')
        data_pre_s, label_pre = get_series_data(data_train, duration, False, obj=[obj])
        duration_pre = len(data_pre_s) * Dt
        Time_array_pre = TimedArray(data_pre_s, dt=Dt)
        stimulus = Time_array_pre
        for loop in range(pre_train_loop):
            net.run(duration_pre)
            net.store('second')
            net.restore('first')
            S4.w = net._stored_state['second']['synapses_3']['w'][0]
            S5.w = net._stored_state['second']['synapses_4']['w'][0]

        # -------change the synapse model----------
        S5.pre.code = S4.pre.code = '''
        h+=w
        g+=w
        '''
        S5.post.code = S4.post.code = ''

        ###############################################
        # ------run for lms_train-------
        del stimulus
        stimulus = Time_array_train
        net.store('third')
        net.run(duration_train)

        # ------lms_train---------------
        y_train = label_to_obj(label_train, obj)
        states, _t_m = get_states(m_read.I, duration * Dt, duration_train, sample)
        Data, para = readout(states, y_train)
        y_train_ = lms_test(states, para)
        y_train_ = normalization_min_max(y_train_)

        #####################################
        # ----run for test--------
        del stimulus
        stimulus = Time_array_test
        net.restore('third')
        net.run(duration_test)

        # -----lms_test-----------
        y_test = label_to_obj(label_test, obj)
        states, t_m = get_states(m_read.I, duration * Dt, duration_test, sample)
        y_test_ = lms_test(states, para)
        y_test_ = normalization_min_max(y_test_)

        roc_auc_train, thresholds_train = ROC(y_train, y_train_)
        print('ROC of train is %s for classification of %s' % (roc_auc_train, obj))
        roc_auc_test, thresholds_test = ROC(y_test, y_test_)
        print('ROC of test is %s for classification of %s' % (roc_auc_test, obj))
        auc_train.append(roc_auc_train)
        auc_test.append(roc_auc_test)
    return [auc_train, auc_test]

if __name__ == '__main__':
    tries = 2
    p = Pool(tries)
    result = p.map(simulate_LSM, np.arange(tries))
    sta_data_tri, sta_data_test = [x[0] for x in result], [x[1] for x in result]
    print(sta_data_test,sta_data_tri)

    # ------vis of results----
    fig_tri = plt.figure(figsize=(10, 4))
    df = pd.DataFrame(np.asarray(sta_data_tri))
    df.boxplot()
    plt.title('Classification Condition of train')

    fig_test = plt.figure(figsize=(10, 4))
    df = pd.DataFrame(np.asarray(sta_data_test))
    df.boxplot()
    plt.title('Classification Condition of test')
    show()
