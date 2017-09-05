from brian2 import *

prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)

#------define function------------
def binary_classification(duration, start=1, end =7, neu =1, interval_l=5, interval_s = ms):
    def tran_bin(A):
        trans = []
        for a in A:
            for i in range(2):
                trans.append(0)
            a_ = bin(a)[2:]
            while len(a_) <3:
                a_ = '0'+a_
            for i in a_:
                trans.append(int(i))
            for i in range(3):
                trans.append(0)
        return np.asarray(trans)
    n = int((duration/interval_s)/interval_l)
    label = np.random.randint(start,end,n)
    seq = tran_bin(label)
    times = where(seq ==1)[0]*interval_s
    indices = zeros(int(len(times)))
    P = SpikeGeneratorGroup(neu, indices, times)
    return P , label

#-----parameter and model setting-------
n = 4
duration = 1000 * ms
interval_l = 8
interval_s = ms
threshold = 0.65
obj = 2

taupre = taupost = 2*ms
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.2

equ = '''
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = (g-h)*40 : 1
'''

equ_1 = '''
dg/dt = (-g)/(1.5*ms) : 1
dh/dt = (-h)/(1.45*ms) : 1
I = (g-h)*20 : 1
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
P, label = binary_classification(duration, start= 3, end=4)
G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=10 * ms, name = 'neurongroup')
G2 = NeuronGroup(round(n/4), equ, threshold='v > 0.30', reset='v = 0', method='linear', refractory=10 * ms, name = 'neurongroup_1')
# G_readout = NeuronGroup(n,equ_1,method='linear')

# S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post= on_post_STDP, method='linear', name = 'synapses')
S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post = on_post_STDP, method='linear', name = 'synapses')
# S3 = Synapses(P, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_2')

S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_1')
S5 = Synapses(G, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_4')

S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post = on_post_STDP, method='linear',  name = 'synapses_3')
# S6 = Synapses(G2, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_5')
# S_readout=Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')
# S4 = Synapses(G, G,'w : 1', on_pre=on_pre, method='linear',  name = 'synapses_3')

#-------network topology----------
S.connect(p = 0.5)
S2.connect()
# S3.connect()
S4.connect(p = 0.7)
S5.connect()

S.w = 'rand()'
S2.w = '-rand()'
# S3.w = '0.3+j*0.2'
S4.w = 'rand()'
S5.w = 'rand()'

#------monitor----------------
# m1 = StateMonitor(G_readout, ('I'), record=True, dt = interval_l*interval_s)
m_g = StateMonitor(G,['v','I'],record=True)
m_w = StateMonitor(S, 'w', record=True)
m_w2 = StateMonitor(S4, 'w', record=True)
# m_s = SpikeMonitor(P)

#------run for pre-train----------------
net = Network(collect())
net.store('first')
net.run(duration)

#------plot the weight----------------
fig1 = plt.figure(figsize= (20,8))
subplot(211)
plot(m_g.t/ms,m_g.I.T)
ylabel('I')
subplot(212)
plot(m_g.t/ms,m_g.v.T)
ylabel('v')

fig2 = plt.figure(figsize= (20,8))
subplot(211)
plot(m_w.t/second, m_w.w.T)
xlabel('Time (s)')
ylabel('Weight / gmax')
subplot(212)
plot(m_w2.t/second, m_w2.w.T)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
show()
