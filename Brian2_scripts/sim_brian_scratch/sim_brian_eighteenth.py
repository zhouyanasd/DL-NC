#------------------------------------------
# special structure RC for STDP competition test
# simulation 6 -- analysis 2
#------------------------------------------
from brian2 import *

prefs.codegen.target = "numpy"
start_scope()
np.random.seed(103)

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

taupre = taupost = 2*ms
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.2

equ = '''
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
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
P, label = binary_classification(duration, start= 6, end=7)
G = NeuronGroup(n, equ, threshold='v > 0.10', reset='v = 0', method='euler', refractory=10 * ms, name = 'neurongroup')
G2 = NeuronGroup(round(n/4), equ, threshold='v > 0.10', reset='v = 0', method='euler', refractory=10 * ms, name = 'neurongroup_1')

S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post = on_post_STDP, method='linear', name = 'synapses')
# S3 = Synapses(P, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_2')

S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_1')
S5 = Synapses(G, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_4')

S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post = on_post_STDP, method='linear',  name = 'synapses_3')
# S6 = Synapses(G2, G2, 'w : 1', on_pre=on_pre, method='linear', name = 'synapses_5')
# S4 = Synapses(G, G,'w : 1', on_pre=on_pre, method='linear',  name = 'synapses_3')

#-------network topology----------
S.connect()
S2.connect()
# S3.connect()
S4.connect(p = 0.7,condition='i != j')
S5.connect()

S.w = 'rand()'
S2.w = '-1'
# S3.w = '0.3+j*0.2'
S4.w = 'rand()'
S5.w = '1'

#------monitor----------------
m_g = StateMonitor(G,['v','I'],record=True)
m_w = StateMonitor(S, 'w', record=True)
m_w2 = StateMonitor(S4, 'w', record=True)
mon_s = SpikeMonitor(P)

#------run for pre-train----------------
net = Network(collect())
net.store('first')
net.run(duration)

#------plot the weight----------------
fig0 = plt.figure(figsize=(20, 4))
plot(mon_s.t/ms, mon_s.i, '.k')
ylim(-0.5,1.5)

fig1 = plt.figure(figsize= (20,8))
subplot(211)
plot(m_g.t/ms,m_g.I.T, label = 'I')
legend(labels = [ ('I_%s'%k) for k in range(n)], loc = 'upper right')
subplot(212)
plot(m_g.t/ms,m_g.v.T, label = 'v')
legend(labels = [ ('V_%s'%k) for k in range(n)], loc = 'upper right')

fig2 = plt.figure(figsize= (20,8))
subplot(211)
plot(m_w.t/second, m_w.w.T, label = 'w')
xlabel('Time (s)')
ylabel('Weight / gmax')
legend(labels = [('from %s to %s w=%s') %(s[0],s[1],s[2])
                 for s in np.vstack((S._synaptic_pre, S._synaptic_post, S.w)).T],
       loc = 'best')
subplot(212)
plot(m_w2.t/second, m_w2.w.T)
xlabel('Time (s)')
ylabel('Weight / gmax')
legend(labels = [('from %s to %s w=%s') %(s[0],s[1],s[2])
                 for s in np.vstack((S4._synaptic_pre, S4._synaptic_post, S4.w)).T],
       loc = 'best')
tight_layout()
show()
