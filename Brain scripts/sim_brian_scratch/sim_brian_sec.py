from brian2 import *
start_scope()

#------define function------------
def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

#-----parameter setting-------
n1 =2
n2 =2

duration = 100*ms
tau = 10*ms
eqs_e = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''

eqs_i = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''

#-----simulation setting-------
P = PoissonGroup(1, 100*Hz)

exc = NeuronGroup(n1, eqs_e, threshold='v > 0.9*mV', reset='v = 0*mV',
                    refractory=1*ms, method='linear')

inh = NeuronGroup(n2, eqs_i, threshold='v > 0.9*mV', reset='v = 0*mV',
                    refractory=1*ms, method='linear')

exc.v = 0*mV
exc.v0 = '0.5*(1+1)*mV'

inh.v = 0*mV
inh.v0 = '0.4*(1+1)*mV'

S_ei = Synapses(exc, inh, on_pre='v_post += 0.5*mV')
# S_ee = Synapses(exc, exc, on_pre='v_post += 0.5*mV')
S_ii = Synapses(inh, inh, on_pre='v_post -= 1*mV')
S_ie = Synapses(exc, exc, on_pre='v_post -= 1.2*mV')

S_input = Synapses(P, exc, on_pre='v+=0.3*mV')

#-------network topology----------
S_ei.connect(j='k for k in range(n2)')
S_ie.connect(j='k for k in range(n1)')
S_ii.connect(j='i')
# S_ee.connect(j='k for k in range(n1) if i!=k')
S_input.connect(j = 'k for k in range(n1)')

#------run----------------
monitor_s_e = SpikeMonitor(exc)
monitor_st_e = StateMonitor(exc, 'v', record=True)
monitor_s_i = SpikeMonitor(inh)
monitor_st_i = StateMonitor(inh, 'v', record=True)

run(duration)

#------vis----------------
visualise_connectivity(S_ei)

fig1 = plt.figure(figsize=(20,4))
subplot(141)
plot(monitor_st_e.t/ms, monitor_st_e.v[0]/mV)
xlabel('Time (ms)')
ylabel('v_e0')
subplot(142)
plot(monitor_st_e.t/ms, monitor_st_e.v[1]/mV)
xlabel('Time (ms)')
ylabel('v_e1')
subplot(143)
plot(monitor_st_i.t/ms, monitor_st_i.v[0]/mV)
xlabel('Time (ms)')
ylabel('v_i0')
subplot(144)
plot(monitor_st_i.t/ms, monitor_st_i.v[1]/mV)
xlabel('Time (ms)')
ylabel('v_i1')

fig5 = plt.figure()
subplot(211)
plot(monitor_s_e.t/ms, monitor_s_e.i, '.k')
plt.ylim(-0.5,1.5)
subplot(212)
plot(monitor_s_i.t/ms, monitor_s_i.i, '.k')
plt.ylim(-0.5,1.5)
show()