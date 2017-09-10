# ----------------------------------------
# correlated and uncorrelated input test for STDP competition
# simulation 6 -- analysis 1
#-----------------------------------------
from brian2 import *

prefs.codegen.target = "numpy"
start_scope()

#------define function------------
def correlated_data(spike_n, neu = 2, interval_l=20,interval_s = 2* ms):
    def tran_correlated(A):
        trans = []
        for a in A:
            trans = np.append(trans,[0]*a)
            trans = np.append(trans, [1]*spike_n)
            if a+spike_n<=interval_l:
                trans = np.append(trans, [0]*(interval_l-a-spike_n))
        return np.asarray(trans)

    def tran_uncorrelated():
        trans = np.array([])
        for a in range(n):
            import random
            s = random.sample(range(interval_l), spike_n)
            trans = np.append(trans, np.asarray(s)+(a*interval_l))
        return np.asarray(trans)

    n = int((duration/ interval_s) / interval_l)
    if spike_n >int(interval_l/2):
        print('too much spike')
    else:
        start = np.random.randint(0, interval_l - spike_n, n)
        seq = tran_correlated(start)
        times_a = where(seq == 1)[0]
        indices_a = zeros(int(len(times_a)))

        times_b = tran_uncorrelated()
        indices_b = zeros(int(len(times_a))) +1

        indices = np.concatenate((indices_a, indices_b), axis=0)
        times = np.concatenate((times_a, times_b), axis=0) * interval_s

        P = SpikeGeneratorGroup(neu, indices, times)
        return P

#------------------------------------------------------------
duration = 1000*ms

taupre = taupost = 15*ms
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.5

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

# P = PoissonGroup(2, rates = np.array([15,35])*Hz)
P = correlated_data(3)

G = NeuronGroup(1, equ, threshold='v > 0.20', reset='v = 0', method='linear', refractory=0 * ms, name = 'neurongroup')

S = Synapses(P, G, model_STDP, on_pre=on_pre_STDP, on_post= on_post_STDP, method='linear', name = 'synapses')

# S.connect(i=0, j=1)
# S.connect(i=1, j=1)
S.connect()

# S.w = '0.3+i*'+str(0.3)
S.w = [0.4,0.6]

M = StateMonitor(S, ['w', 'apre', 'apost'], record=True)
M_G = StateMonitor(G, ['v','I'], record = True)
mon_s = SpikeMonitor(P)

net = Network(collect())
net.store('first')
net.run(duration)

fig0 = plt.figure(figsize=(20, 4))
plot(mon_s.t/ms, mon_s.i, '.k')
ylim(-0.5,1.5)

fig1 = figure(figsize=(20, 8))
subplot(211)
plot(M.t/ms, M.apre[0], label='apre')
plot(M.t/ms, M.apost[0], label='apost')
legend()
subplot(212)
plot(M.t/ms, M.w.T, label='w')
legend(loc='best')
xlabel('Time (ms)')

fig2 = figure(figsize=(20, 8))
subplot(211)
plot(M_G.t/ms, M_G.v[0], label='v 1')
legend()
subplot(212)
plot(M_G.t/ms, M_G.I[0], label='I 1')
legend()
show()