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

stimulus = TimedArray(np.hstack([[c, c, c, 0, 0]
                                 for c in np.random.rand(1000)]),
                                dt=10*ms)
G = NeuronGroup(10, 'dv/dt = (-v + stimulus(t))/(10*ms) : 1',
                threshold='v>1', reset='v=0')
G.v = '0.5*rand()'  # different initial values for the neurons

statemon = StateMonitor(G, 'v', record=0)

plot(statemon.t/ms, statemon.v[0])

show()