#----------------------------------------
# inhibitory neuron WTA and digit number classification test
# multiple pre-train STDP
# simulation 6--analysis 4
#----------------------------------------

from brian2 import *
from brian2tools import *
from scipy.optimize import leastsq
import scipy as sp

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
    Data = []
    for i in M:
        Data.append(i[1:])
    l = len(p)
    f = p[l - 1]
    for i in range(len(Data)):
        f += p[i] * Data[i]
    return f

def readout(M, Z):
    n = len(M)
    Data = []
    for i in M:
        Data.append(i[1:])
    p0 = [1]*n
    p0.append(0.1)
    para = lms_train(p0, Z, Data)
    return Data,para

def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def binary_classification(duration, start=1, end =4, neu =1, interval_l=10, interval_s = ms):
    def tran_bin(A):
        trans = []
        for a in A:
            for i in range(3):
                trans.append(0)
            a_ = bin(a)[2:]
            while len(a_) <3:
                a_ = '0'+a_
            for i in a_:
                trans.append(int(i))
            for i in range(4):
                trans.append(0)
        return np.asarray(trans)
    n = int((duration/interval_s)/interval_l)
    label = np.random.randint(start,end,n)
    seq = tran_bin(label)
    times = where(seq ==1)[0]*interval_s
    indices = zeros(int(len(times)))
    P = SpikeGeneratorGroup(neu, indices, times)
    return P , label

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
            x = float(arr[i] - np.min(arr))/(np.max(arr) - np.min(arr))
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
    return fig, roc_auc , thresholds


###############################################
#-----parameter and model setting-------
n = 4
pre_train_duration = 4000*ms
duration = 500 * ms
duration_test = 100*ms
pre_train_loop = 5
interval_l = 10
interval_s = ms
threshold = 0.4
obj = 3

t0 = int(duration/ (interval_l*interval_s))
t1 = int((duration+duration_test) / (interval_l*interval_s))

taupre = taupost = 2*ms
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.2

equ = '''
dv/dt = (I-v) / (2*ms) : 1 (unless refractory)
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = tanh(g-h)*30 : 1
'''

equ_1 = '''
dg/dt = (-g)/(1.5*ms) : 1 
dh/dt = (-h)/(1.45*ms) : 1
I = tanh(g-h)*20 : 1
'''

on_pre = '''
h+=w
g+=w
'''

model_STDP= '''
w : 1
dapre/dt = -apre/taupre : 1 (clock-driven)
dapost/dt = -apost/taupost : 1 (clock-driven)
'''

on_pre_STDP = '''
h+=w
g+=w
apre += Apre
w = clip(w+apost, 0, wmax)
'''

on_post_STDP= '''
apost += Apost
w = clip(w+apre, 0, wmax)
'''

#-----simulation setting-------
P_plasticity, label_plasticity = binary_classification(pre_train_duration, start=3,end=4,
                                                       interval_l=interval_l,interval_s = interval_s)
P, label = binary_classification(duration + duration_test, start=1,end=7,
                                 interval_l=interval_l,interval_s = interval_s)
G = NeuronGroup(n, equ, threshold='v > 0.15', reset='v = 0', method='euler', refractory=3 * ms,
                name = 'neurongroup')
G2 = NeuronGroup(round(n/4), equ, threshold ='v > 0.15', reset='v = 0', method='euler', refractory=2 * ms,
                 name = 'neurongroup_1')
G_readout = NeuronGroup(n,equ_1, method ='euler')

# S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post= on_post_STDP, method='linear', name = 'synapses')
S = Synapses(P_plasticity, G,'w : 1', on_pre = on_pre, method='linear', name = 'synapses')
# S3 = Synapses(P, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_2')

S2 = Synapses(G2, G, 'w : 1', on_pre = on_pre, method='linear', name = 'synapses_1')
S5 = Synapses(G, G2, 'w : 1', on_pre = on_pre, method='linear', name = 'synapses_4')

S4 = Synapses(G, G, model_STDP, on_pre = on_pre_STDP, on_post = on_post_STDP, method = 'linear',  name = 'synapses_3')
# S6 = Synapses(G2, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_1')
S_readout = Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')
# S4 = Synapses(G, G,'w : 1', on_pre=on_pre, method='linear',  name = 'synapses_3')

#-------network topology----------
S.connect(j='k for k in range(n)')
S2.connect()
S4.connect(condition='i != j')
S5.connect()
S_readout.connect(j='i')

S.w = '0.1+j*'+str(0.9/n)
S2.w = '-rand()'
S4.w = 'rand()'
S5.w = 'rand()'

#------monitor----------------
m1 = StateMonitor(G_readout, ('I'), record=True, dt = interval_l*interval_s)
m_w = StateMonitor(S, 'w', record=True)
m_w2 = StateMonitor(S4, 'w', record=True)
m_s = SpikeMonitor(P)
m_g = StateMonitor(G, (['I','v']), record = True)
m_g2 = StateMonitor(G2, (['I','v']), record = True)
m_read = StateMonitor(G_readout, ('I'), record = True)

#------create network-------------
net = Network(collect())
net.store('first')
print('S4.w = %s'%S4.w)
###############################################
#------pre_train------------------
for loop in range(pre_train_loop):
    net.run(pre_train_duration)

    # ------plot the weight----------------
    fig2 = plt.figure(figsize=(10, 8))
    title('loop: '+str(loop))
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
    S.w = net._stored_state['second']['synapses']['w'][0]

net.remove(P_plasticity)
S.source = P
S.pre.source = P
S._dependencies.remove(P_plasticity.id)
S.add_dependency(P)
S.connect(j='k for k in range(n)')

#-------change the synapse model----------
S4.pre.code = '''
h+=w
g+=w
'''
S4.post.code = ''

###############################################
#------run for lms_train-------
net.store('third')
net.run(duration, report='text')

#------lms_train---------------
y = label_to_obj(label[:t0],obj)
m1.record_single_timestep()
Data,para = readout(m1.I,y)

#####################################
#----run for test--------
net.restore('third')
net.run(duration+duration_test, report='text')

#-----lms_test-----------
obj_t = label_to_obj(label,obj)
m1.record_single_timestep()
y_t = lms_test(m1.I, para)

y_t_class, data_n = classification(threshold, y_t)
fig_roc_train, roc_auc_train , thresholds_train = ROC(obj_t[:t0],data_n[:t0],'ROC for train')
print('ROC of train is %s'%roc_auc_train)
fig_roc_test, roc_auc_test , thresholds_test = ROC(obj_t[t0:],data_n[t0:],'ROC for test')
print('ROC of test is %s'%roc_auc_test)

#####################################
#------vis of results----
fig1 = plt.figure(figsize=(20, 8))
subplot(211)
plt.scatter(m1.t[1:] / ms, y_t_class, s=2, color="red", marker='o', alpha=0.6)
plt.scatter(m1.t[1:] / ms, obj_t, s=3, color="blue", marker='*', alpha=0.4)
plt.scatter(m1.t[1:] / ms, data_n, color="green")
axhline(threshold, ls='--', c='r', lw=1)
axvline(duration/ms, ls='--', c='green', lw=3)
subplot(212)
plot(m_s.t/ms, m_s.i, '.k')
ylim(-0.5,0.5)

fig3 = plt.figure(figsize=(20,8))
subplot(211)
plt.plot(m_g.t / ms, m_g.v.T,label='v')
legend(labels = [ ('V_%s'%k) for k in range(n)], loc = 'upper right')
subplot(212)
plt.plot(m_g.t / ms, m_g.I.T,label='I')
legend(labels = [ ('I_%s'%k) for k in range(n)], loc = 'upper right')

fig4 = plt.figure(figsize=(20,8))
subplot(211)
plt.plot(m_g2.t / ms, m_g2.v.T,label='v')
legend(labels = [ ('V_%s'%k) for k in range(n)], loc = 'upper right')
subplot(212)
plt.plot(m_g2.t / ms, m_g2.I.T,label='I')
legend(labels = [ ('I_%s'%k) for k in range(n)], loc = 'upper right')

fig5 = plt.figure(figsize=(20,4))
plt.plot(m_read.t / ms, m_read.I.T,label='I')
legend(labels = [ ('I_%s'%k) for k in range(n)], loc = 'upper right')

fig6 =plt.figure(figsize=(4,4))
brian_plot(S4.w)
show()
