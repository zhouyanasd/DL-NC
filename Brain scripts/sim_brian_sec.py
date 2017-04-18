from brian2 import *
start_scope()

n1 =2
n2 =2

duration = 10*ms
tau = 10*ms
eqs_e = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''

eqs_i = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''

exc = NeuronGroup(n1, eqs_e, threshold='v > 3*mV', reset='v = 0*mV',
                    refractory=1*ms, method='linear')

inh = NeuronGroup(n2, eqs_i, threshold='v > 3*mV', reset='v = 0*mV',
                    refractory=1*ms, method='linear')

exc.v = 0*mV
exc.v0 = '20*mV * i / (n1-1)'

inh.v = 0*mV
inh.v0 = '10*mV'

monitor_s_e = SpikeMonitor(exc)
monitor_st_e = StateMonitor(exc, 'v', record=0)
monitor_s_i = SpikeMonitor(inh)
monitor_st_i = StateMonitor(inh, 'v', record=0)

run(duration)
# fig1 = plt.figure()
# plot(monitor_s_e.v0/mV, monitor_s_e.count / duration)
# xlabel('v0 (mV)')
# ylabel('Firing rate (sp/s)')
# fig2 = plt.figure()
# plot(monitor_s_i.v0/mV, monitor_s_i.count / duration)
# xlabel('v0 (mV)')
# ylabel('Firing rate (sp/s)')
fig3 = plt.figure()
plot(monitor_st_e.t/ms, monitor_st_e.v[0])
xlabel('Time (ms)')
ylabel('v')
fig4 = plt.figure()
plot(monitor_st_i.t/ms, monitor_st_i.v[0])
xlabel('Time (ms)')
ylabel('v')
show()