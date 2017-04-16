from brian2 import *
start_scope()

# tau = 5*ms
# eqs = '''
# dv/dt = (1-v)/tau : 1
# '''
#
# G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', refractory=15*ms, method='linear')
#
# statemon = StateMonitor(G, 'v', record=0)
# spikemon = SpikeMonitor(G)
#
# run(50*ms)
#
# plot(statemon.t/ms, statemon.v[0])
# for t in spikemon.t:
#     axvline(t/ms, ls='--', c='r', lw=3)
# axhline(0.8, ls=':', c='g', lw=3)
# xlabel('Time (ms)')
# ylabel('v')
# print("Spike times: %s" % spikemon.t[:])
# show()

tau_pre = tau_post = 20*ms
A_pre = 0.01
A_post = -A_pre*1.05
delta_t = linspace(-50, 50, 100)*ms
W = where(delta_t<0, A_pre*exp(delta_t/tau_pre), A_post*exp(-delta_t/tau_post))

plot(delta_t / ms, W)

xlabel(r'$\Delta t$ (ms)')
ylabel('W')
ylim(-A_post, A_post)
axhline(0, ls='-', c='k')

show()