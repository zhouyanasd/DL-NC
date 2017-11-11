#----------------------------------------
# Results for Tri-function in Feedforword without STDP
#----------------------------------------

from brian2 import *
from scipy.optimize import leastsq
import scipy as sp
import pandas as pd

prefs.codegen.target = "numpy"  #it is faster than use default "cython"
start_scope()
np.random.seed(101)

#------define function------------
def lms_train(p0,Zi,Data):
    def error(p, y, args):
        l = len(p)
        f = p[l - 1]
        for i in range(len(args)):
            f += p[i] * args[i]
        return f - y
    Para = leastsq(error,p0,args=(Zi,Data))
    return Para[0]

def lms_test(M, p):
    l = len(p)
    f = p[l - 1]
    for i in range(len(M)):
        f += p[i] * M[i]
    return f

def readout(M,Z):
    n = len(M)
    Data=[]
    for i in M:
        Data.append(i)
    p0 = [1]*n
    p0.append(0.1)
    para = lms_train(p0, Z, Data)
    return Data,para

def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))


def Tri_function(duration):
    rng = np.random
    TIME_SCALE = defaultclock.dt
    in_number = int(duration/TIME_SCALE)

    def sin_fun(l, c, t):
        return (np.sin(c * t * TIME_SCALE/us) + 1) / 2

    def tent_map(l, c, t):
        temp = l
        if (temp < 0.5 and temp > 0):
            temp = (c / 101+1) * temp
            return temp
        elif (temp >= 0.5 and temp < 1):
            temp = (c / 101+1) * (1 - temp)
            return temp
        else:
            return 0.5

    def constant(l, c, t):
        return c / 100

    def chose_fun():
        c = rng.randint(0, 3)
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
        if change_fun(0.7) and t % 50 ==0:
            cons = rng.randint(1, 101)
            fun, c = chose_fun()
            try:
                data_t= fun(data[t - 1], cons, t)
                data.append(data_t)
                cla.append(c)
            except IndexError:
                data_t = fun(rng.randint(1, 101)/100, cons, t)
                data.append(data_t)
                cla.append(c)
        else:
            try:
                data_t = fun(data[t - 1], cons, t)
                data.append(data_t)
                cla.append(c)
            except IndexError:
                data_t= fun(rng.randint(1, 101)/100, cons, t)
                data.append(data_t)
                cla.append(c)
    cla = np.asarray(cla)
    data = np.asarray(data)
    return data, cla

def label_to_obj(label,obj):
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
            x = float(arr[i] - np.min(arr))/(np.max(arr)- np.min(arr))
            arr_n[i] = x
        return arr_n
    data_n = normalization_min_max(data)
    data_class = []
    for a in data_n:
        if a >=thea:
            b = 1
        else:
            b = 0
        data_class.append(b)
    return np.asarray(data_class),data_n

def ROC(y, scores, fig_title = 'ROC', pos_label=1):
    def normalization_min_max(arr):
        arr_n = arr
        for i in range(arr.size):
            x = float(arr[i] - np.min(arr))/(np.max(arr) - np.min(arr))
            arr_n[i] = x
        return arr_n
    scores_n = normalization_min_max(scores)
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y, scores_n, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)

    # fig = plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(fig_title)
    # plt.legend(loc="lower right")
    # return fig, roc_auc , thresholds
    # return fig, roc_auc, thresholds
    return roc_auc, thresholds

#-----parameter setting-------
loop = 5
sta_data_tri = []
sta_data_test = []
for l in range(loop):
    np.random.seed(l)

    n = 20
    duration = 200 * ms
    duration_test = 200*ms
    interval_s = ms
    threshold = 0.5
    obj = 0

    equ = '''
    dv/dt = (I-v) / (0.3*ms) : 1 (unless refractory)
    dg/dt = (-g)/(0.15*ms) : 1
    dh/dt = (-h)/(0.145*ms) : 1
    I = (g-h)*40 +I_0: 1
    I_0 = stimulus(t)*w_g:1
    w_g : 1
    '''

    equ_1 = '''
    dg/dt = (-g)/(1.5*ms) : 1
    dh/dt = (-h)/(1.45*ms) : 1
    I = (g-h)*30 : 1
    '''

    on_pre = '''
    h+=w
    g+=w
    '''

    #-----simulation setting-------

    data , label = Tri_function(duration+duration_test)
    stimulus = TimedArray(data,dt=defaultclock.dt)

    G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms)
    G2 = NeuronGroup(2, equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms)
    G_readout=NeuronGroup(n,equ_1,method='linear')

    S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
    S4 = Synapses(G, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
    S_readout=Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

    #-------network topology----------
    S2.connect()
    S4.connect(condition='i != j', p=0.1)
    S_readout.connect(j='i')


    G.w_g = '0.2+i*'+str(0.8/n)
    G2.w_g = '0.3+i*0.3'

    S2.w = '-rand()/2'
    S4.w = 'rand()'

    #------run----------------
    m3 = StateMonitor(G_readout, ('I'), record=True)
    m4 = StateMonitor(G, ('I'), record=True)

    net = Network(collect())
    net.store('first')
    auc_test = []
    auc_train = []
    for epochs in range(3):
        obj = epochs
        net.restore('first')
        net.run(duration)

        #----lms_readout----#
        t0 = int(duration/defaultclock.dt)
        obj1 = label_to_obj(label,obj)
        Data,para = readout(m3.I,obj1[:t0])

        #----lms_test-------#
        net.restore('first')
        net.run(duration+duration_test)
        obj1_t = lms_test(m3.I,para)
        obj1_t_class,data_n = classification(threshold,obj1_t)
        # fig_roc_train, roc_auc_train, thresholds_train = ROC(obj1[:t0], data_n[:t0], 'ROC for train of %s' % obj)
        roc_auc_train , thresholds_train = ROC(obj1[:t0], data_n[:t0],'ROC for train of %s'%obj)
        print('ROC of train is %s for classification of %s'%(roc_auc_train,obj))
        # fig_roc_test, roc_auc_test, thresholds_test = ROC(obj1[t0:], data_n[t0:], 'ROC for test of %s' % obj)
        roc_auc_test , thresholds_test = ROC(obj1[t0:], data_n[t0:],'ROC for test of %s'%obj)
        print('ROC of test is %sfor classification of %s'%(roc_auc_test,obj))

        auc_train.append(roc_auc_train)
        auc_test.append(roc_auc_test)
    sta_data_tri.append(auc_train)
    sta_data_test.append(auc_test)
        # fig1 = plt.figure(figsize=(20, 4))
        # subplot(111)
        # plt.scatter(m3.t / ms, obj1_t_class,s=2, color="red", marker='o',alpha=0.6)
        # plt.scatter(m3.t / ms, obj1,s=3,color="blue",marker='*',alpha=0.4)
        # plt.scatter(m3.t / ms, data_n,s=2,color="green")
        # axhline(threshold, ls='--', c='r', lw=1)
        # plt.title('Classification Condition of threshold = 0.5')
        #
    #------vis----------------
    # fig0 = plt.figure(figsize=(20, 4))
    # plot(data, 'r')
    # plt.title('Orange Signal')

fig_tri = plt.figure(figsize=(4, 4))
df = pd.DataFrame(np.asarray(sta_data_tri),
columns=['0', '1', '3'])
df.boxplot()
plt.title('Classification Condition of train')

fig_test = plt.figure(figsize=(4, 4))
df = pd.DataFrame(np.asarray(sta_data_test),
columns=['0', '1', '3'])
df.boxplot()
plt.title('Classification Condition of test')
show()
