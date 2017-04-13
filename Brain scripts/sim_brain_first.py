from brian2 import *
start_scope()

tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''

G = NeuronGroup(1, eqs, method='linear')
M = StateMonitor(G, 'v', record=0)

run(100*ms)

print('After v = %s' % G.v[0])

plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v')