from brian2 import *
from scipy.optimize import leastsq
import scipy as sp

start_scope()
np.random.seed(100)

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

def lms_test(Data, p):
    l = len(p)
    f = p[l - 1]
    for i in range(len(Data)):
        f += p[i] * Data[i]
    return f

def readout(M):
    n = len(M)
    Data=[]
    for i in range(n):
        x = M[i].smooth_rate(window='gaussian', width=time_window)/ Hz
        Data.append(x)
    return Data

def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def save_para(para, name):
    np.save('../Data/'+str(name)+'.npy',para)

def load_para(name):
    return np.load('../Data/'+str(name)+'.npy')
#---loop-----

N = 40
MSE_train = []
MSE_test = []

for n in range(1,N):
    print('loop: ', n)
    # -----parameter setting-------
    time_window = 10 * ms
    duration = 5000* ms
    duration_test = 2000 * ms

    equ = '''
    dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
    dg/dt = (-g)/(10*ms) : 1
    dh/dt = (-h)/(9.5*ms) : 1
    I = (g-h)*20 : 1
    '''

    on_pre = '''
    h+=w
    g+=w
    '''

    # -----simulation setting-------
    P = PoissonGroup(1, 50 * Hz)
    G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms)
    G2 = NeuronGroup(2, equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=0 * ms)
    S = Synapses(P, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
    S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.5 * ms)
    S3 = Synapses(P, G2, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)
    S4 = Synapses(G, G, 'w : 1', on_pre=on_pre, method='linear', delay=0.1 * ms)

    # -------network topology----------
    S.connect(j='k for k in range(n)')
    S2.connect()
    S3.connect()
    S4.connect(condition='i != j', p=0.1)

    S.w = '0.1+j*' + str(1 / n)
    S2.w = '-rand()/2'
    S3.w = '0.3+j*0.3'
    S4.w = 'rand()'

    # ------monitor----------------
    M = []
    for mi in range(G._N):
        locals()['M' + str(mi)] = PopulationRateMonitor(G[(mi):(mi + 1)])
        M.append(locals()['M' + str(mi)])
    m_y = PopulationRateMonitor(P)

    # ------run for train----------------
    run(duration)

    # ----state_readout-----
    Data = readout(M)
    Y = (m_y.smooth_rate(window='gaussian', width=time_window) / Hz)

    # ----lms_train------
    p0 = [1] * n
    p0.append(0.1)
    para = lms_train(p0, Y, Data)
    save_para(para, 'para_readout_eighth_'+str(n))
    print(load_para('para_readout_eighth_'+str(n)))

    Y_t = lms_test(Data, para)
    MSE_train.append(mse(Y_t,Y))
    # ----run for test--------
    run(duration_test)

    # -----test_Data----------
    Data = readout(M)
    Y = (m_y.smooth_rate(window='gaussian', width=time_window) / Hz)

    # -----lms_test-----------
    Y_t = lms_test(Data, para)
    t0 = int(duration / defaultclock.dt)
    t1 = int((duration + duration_test) / defaultclock.dt)

    Y_test = Y[t0:t1]
    Y_test_t = Y_t[t0:t1]
    MSE_test.append(mse(Y_test_t,Y_test))

    #-----stop----------------
    stop()
    start_scope()

#-----vis-------
t_mse = np.arange(1,N)
fig1 = plt.figure(figsize=(20, 4))
subplot(111)
plot(t_mse , MSE_train,'-b',label = "train")
plot(t_mse , MSE_test,'--r',label = "test")
xlabel('number of neuron')
ylabel('mse')
legend()
show()
