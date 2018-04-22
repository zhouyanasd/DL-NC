from brian2 import *

start_scope()
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms

equ = '''
dv/dt = (I-v) / (30*ms) : 1 (unless refractory)
dh/dt = (-h)/(6*ms) : 1
I = (h)+13.5: 1
'''

on_pre = '''
h+=w
'''
stimulus = TimedArray(np.tile([20., 0.], 5)*Hz, dt=1000.*ms)
P = PoissonGroup(1, rates='stimulus(t)')

G = NeuronGroup(10, equ, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                name='neurongroup_ex')
S_in = Synapses(P, G, 'w : 1', on_pre=on_pre, method='euler', name='synapses_in', delay=1.5 * ms)
S_e = Synapses(G, G, 'w : 1', on_pre=on_pre, method='euler', name='synapses_E')
S_in.connect(p=0.3)
S_e.connect(condition='i!=j', p=0.5)

G.v = '13.5+1.5*rand()'
G.h = '0'
S_in.w = 'randn()*30+30'
S_e.w = 'randn()*30+30'
S_e.delay='randn()*3 * ms+3*ms'

# G.run_regularly(''' lastspike = 0 * ms
#                     v = 13.5+1.5*rand()
#                     h = 0
#                     ''', dt=1000 * ms)
@network_operation(dt=1000*ms)
def update_active():
    for pathway in S_e._pathways:
        pathway.queue._restore_from_full_state(None)
    G.lastspike = '0 * ms'
    G.not_refractory = True
    G.v = '13.5+1.5*rand()'
    G.h = '0'
#     S_e._pathways[0].queue.X = zeros(S_e._pathways[0].queue.X.shape).astype(int)
#     S_e._pathways[0].queue.X_flat = zeros(S_e._pathways[0].queue.X_flat.shape).astype(int)
#     S_e._pathways[0].queue.n = zeros(S_e._pathways[0].queue.n.shape).astype(int)
#     S_in._pathways[0].queue.X = zeros(S_in._pathways[0].queue.X.shape).astype(int)
#     S_in._pathways[0].queue.X_flat = zeros(S_in._pathways[0].queue.X_flat.shape).astype(int)
#     S_in._pathways[0].queue.n = zeros(S_in._pathways[0].queue.n.shape).astype(int)

net = Network(collect())
net.store('init')
net.run(1000*ms, report='text')

# net.store('temp')
# net.restore('temp')
#
# net.run(1*ms)
#
# G._full_state()
#
# S_e._full_state()
# S_e._pathways[0].queue.__dict__

# S_e._pathways[0].queue._restore_from_full_state(None)
# S_e._pathways[0].queue._full_state()

# state = net._stored_state['temp']
# state