from brian2 import *

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

def add_group_neuron(net, neuron_group, name = 'agn'):
    # {'_spikespace': (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 16),
    #  'g': (array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #               0., 0.]), 15),
    #  'h': (array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #               0., 0.]), 15),
    #  'i': (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    #        15),
    #  'lastspike': (array([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,
    #                       -inf, -inf, -inf, -inf]), 15),
    #  'not_refractory': (array([True, True, True, True, True, True, True, True, True,
    #                            True, True, True, True, True, True], dtype=bool), 15),
    #  'v': (array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #               0., 0.]), 15)}
    net.store(name)
    state = net._stored_state[name]
    neuron = state[neuron_group.name]
    pass





def delete_group_neuron(net, neuron_group, num, name = 'dgn'):
    pass

def delete_group_synapse(net, synapse_group, source, target, name = 'dgn'):

    # {'N': (array([3]), 1),
    #  'N_incoming': (array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]), 10),
    #  'N_outgoing': (array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]), 10),
    #  '_synaptic_post': (array([9, 8, 7]), 3),
    #  '_synaptic_pre': (array([9, 8, 7]), 3),
    #  'delay': (array([0.009, 0.008, 0.02]), 3),
    #  'lastupdate': (array([0.2932, 0.2962, 0.2878]), 3)}

    def delete_para(index,name):
        para =  list(synapse[name])

    def delete_N_(source,target):
        synapse['N_outgoing'][0][source] = synapse['N_outgoing'][0][source] - 1
        synapse['N_incoming'][0][target] = synapse['N_incoming'][0][target] - 1

    def delete_synaptic(index):
        synaptic_pre =  list(synapse['_synaptic_pre'])
        synaptic_post = list(synapse['_synaptic_post'])
        synaptic_pre[0] = np.delete(synaptic_pre[0], index, axis=0)
        synaptic_pre[1] = synaptic_pre[1] - 1
        synapse['_synaptic_pre'] = tuple (synaptic_pre)
        synaptic_post[0] = np.delete(synaptic_post[0], index, axis=0)
        synaptic_post[1] = synaptic_post[1] -1
        synapse['_synaptic_post'] = tuple(synaptic_post)


    def delete_delay_update(index):
        lastupdate =  list(synapse['lastupdate'])
        delay = list(synapse['delay'])
        lastupdate[0] = np.delete(lastupdate[0], index, axis=0)
        lastupdate[1] = lastupdate[1] - 1
        delay[0] = np.delete(delay[0], index, axis=0)
        delay[1] = delay[1] - 1
        synapse['lastupdate'] = tuple (lastupdate)
        synapse['delay'] = tuple(delay)


    net.store(name)
    state = net._stored_state[name]
    synapse = state[synapse_group.name]
    s_p = np.vstack((synapse['_synaptic_pre'][0],synapse['_synaptic_post'][0])).T
    s = np.hstack((np.array([source]), np.array([target])))
    try:
        index = np.where((s_p == s).all(1))[0][0]
        delete_synaptic(index)
        delete_delay_update(index)
        delete_N_(source,target)
        synapse['N'][0][0] = synapse['N'][0][0] -1
    except IndexError:
        print('No synapse_path %s here.' %s)
    net.restore(name)
    # print(synapse)


#--------------------------------------

start_scope()
np.random.seed(15)

equ = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*40 : 1
'''

on_pre ='''
h+=w
g+=w
'''

P = PoissonGroup(10, np.arange(10)*Hz + 50*Hz)
G = NeuronGroup(10, equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms )
S = Synapses(P, G, 'w : 1',on_pre = on_pre, method='linear')
S.connect(j =8, i= 8)

S.delay = 'j*ms'
S.w = '0.1+j*0.1'

m1 = StateMonitor(G,'v',record=9)
M = StateMonitor(G, 'I', record=9)
m2 = SpikeMonitor(P)

net = Network(collect())
net.run(100*ms)
net.store('first')
visualise_connectivity(S)


G.equations._equations['I'] = "I = (g-h)*30 : 1"
G.equations._equations.pop('I')
G.equations = G.equations+("I = (g-h)*30 : 1")


net.run(100*ms)


fig1 = plt.figure(figsize=(10,4))
subplot(121)
plot(m1.t/ms,m1.v[0],'-b', label='Neuron 9 V')
legend()

subplot(122)
plot(M.t/ms, M.I[0], label='Neuron 9 I')
legend()

show()