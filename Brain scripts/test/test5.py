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

def find_synapse(synapse_group, source, target):
    s_p = np.vstack((synapse_group._synaptic_pre, synapse_group._synaptic_post)).T
    s = np.hstack((np.array([source]), np.array([target])))

    try:
        index = np.where((s_p == s).all(1))[0][0]
        return index
    except IndexError:
        print('No synapse_path %s here.' % s)


def add_para(module, name, p):
    para = list(module[name])
    if p.ndim == 1:
        para[0] = np.append(para[0], p)
        para[1] = para[0].size
    elif p.ndim == 2:
        para[0] = np.append(para[0], p, axis=1)
        para[1] = para[0].shape
    else:
        print('Out of the dim which can be handled')
    module[name] = tuple(para)

def add_neuron_mon(net, mon, neuron_group, index, name = 'amn'):
    net.store(name)
    monitor = net._stored_state[name][mon.name]
    N = mon.n_indices
    mon.n_indices += 1
    if mon.source.name == neuron_group.name:
        if index not in mon.variables['_indices'].get_value():
            mon.variables['_indices'].device.arrays[mon.variables['_indices']] \
                = np.zeros((N + 1), dtype=int32)
            mon.variables['_indices'].size = N + 1
            add_para(monitor, '_indices', np.array([index]))

            t = monitor['N'][0][0]
            m_c = monitor.copy()
            m_c.pop('N')
            m_c.pop('t')
            m_c.pop('_indices')
            for c in m_c:
                mon.variables[c].resize_along_first = False
                mon.variables[c].resize((t, N + 1))
                mon.variables[c].device.arrays[mon.variables[c]]._data = mon.variables[c].get_value()
                mon.variables[c].resize_along_first = True
                add_para(monitor, c, np.zeros((t, 1)))
        else:
            print("The %s neuron has been monitored by statemonitor %s"%(index, mon))
    else:
        print(" the statemoinitor %s is not for neurongroup %s ." % (mon.name, neuron_group))

    net.restore(name)


def add_group_neuron(net, neuron_group, num=1, name='agn'):
    # {'_spikespace': (array([0, 0, 0, 0]), 4),
    #  'g': (array([0., 0., 0.]), 3),
    #  'h': (array([0., 0., 0.]), 3),
    #  'i': (array([0, 1, 2]), 3),
    #  'lastspike': (array([-inf, -inf, -inf]), 3),
    #  'not_refractory': (array([True, True, True], dtype=bool), 3),
    #  'v': (array([0., 0., 0.]), 3)}
    #
    # 'synapses': {  # 'N_incoming': (array([0, 0, 0, 1, 1]), 5),
    # 'N_outgoing': (array([0, 0, 0, 1, 1]), 5)}


    def add_neuron_to_synapse(net,num,name):
        for obj in net.objects:
            if isinstance(obj, Synapses):
                synapse_t = net._stored_state[name][obj.name]
                if obj.source.name == neuron_group.name:
                    obj.variables['N_pre'].value = N + num
                    add_para(synapse_t,'N_outgoing',np.array([0.]*num))
                elif obj.target.name == neuron_group.name:
                    obj.variables['N_post'].value = N + num
                    add_para(synapse_t, 'N_incoming', np.array([0.]*num))



    net.store(name)
    state = net._stored_state[name]
    neuron = state[neuron_group.name]
    N = neuron_group._N

    neuron_group._N = N+num
    neuron_group.stop = N+num
    neuron_group.variables['N'].value = N+num
    neuron_group.variables['lastspike'].device.arrays[ neuron_group.variables['lastspike']] \
        = np.array([-inf]*(N+num))
    neuron_group.variables['lastspike'].size = neuron_group.variables['lastspike'].size +num
    neuron_group.variables['not_refractory'].device.arrays[neuron_group.variables['not_refractory']] \
        = np.array([True ] * (N + num), dtype=bool)
    neuron_group.variables['not_refractory'].size = neuron_group.variables['not_refractory'].size + num
    neuron_group.variables['i'].device.arrays[neuron_group.variables['i']] \
        = np.arange(N+num)
    neuron_group.variables['i'].size = neuron_group.variables['i'].size + num
    neuron_group.variables['_spikespace'].device.arrays[neuron_group.variables['_spikespace']] \
        = np.zeros((N + num+1),dtype=int32)
    neuron_group.variables['_spikespace'].size = neuron_group.variables['_spikespace'].size + num
    # neuron_group._create_variables(None, events=list(neuron_group.events.keys()))

    add_para(neuron,'lastspike',np.array([-inf]*num))
    add_para(neuron,'not_refractory', np.array([True]*num))
    add_para(neuron, 'i', np.arange(N,N+num))
    add_para(neuron,'_spikespace',np.zeros(num,dtype=int32))

    n_c = neuron.copy()
    n_c.pop('lastspike')
    n_c.pop('not_refractory')
    n_c.pop('i')
    n_c.pop('_spikespace')
    for c in n_c:
        neuron_group.variables[c].device.arrays[neuron_group.variables[c]] \
            = np.zeros((N+num),dtype=float64)
        neuron_group.variables[c].size = neuron_group.variables[c].size + num
        add_para(neuron, c, np.array([0.]*num))

    add_neuron_to_synapse(net,num,name)
    # add_neuron_to_mon(net,num,name)

    net.restore(name)


def delete_group_neuron(net, neuron_group, index, name = 'dgn'):

    def delete_connected_synapses(net, neuron_group, index):
        for obj in net.objects:
            if isinstance(obj, Synapses):
                synapse_t = net._stored_state[name][obj.name]
                if obj.source.name == neuron_group.name:
                    index_ = where(synapse_t['_synaptic_pre'][0] == index)[0]
                    for i in index_:
                        delete_group_synapse(net,obj,index,synapse_t['_synaptic_post'][0][i])
                elif obj.target.name == neuron_group.name:
                    index_ = where(synapse_t['_synaptic_post'][0] == index)[0]
                    for i in index_:
                        delete_group_synapse(net,obj,synapse_t['_synaptic_pre'][0][i],index)



    # def delete_connected_mon(net):
    #     for obj in net.objects:
    #         if isinstance(obj, StateMonitor):
    #             mon_t = net._stored_state[name][obj.name]


    net.store(name)
    # state = net._stored_state[name]
    # neuron = state[neuron_group.name]
    delete_connected_synapses(net, neuron_group, index)

    net.restore(name)



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
        para[0] = np.delete(para[0], index, axis=0)
        para[1] = para[1] - 1
        synapse[name] = tuple (para)

    # def delete_N_(source,target):
    #     synapse['N_outgoing'][0][source] = synapse['N_outgoing'][0][source] - 1
    #     synapse['N_incoming'][0][target] = synapse['N_incoming'][0][target] - 1

    # def delete_synaptic(index):
    #     synaptic_pre =  list(synapse['_synaptic_pre'])
    #     synaptic_post = list(synapse['_synaptic_post'])
    #     synaptic_pre[0] = np.delete(synaptic_pre[0], index, axis=0)
    #     synaptic_pre[1] = synaptic_pre[1] - 1
    #     synapse['_synaptic_pre'] = tuple (synaptic_pre)
    #     synaptic_post[0] = np.delete(synaptic_post[0], index, axis=0)
    #     synaptic_post[1] = synaptic_post[1] -1
    #     synapse['_synaptic_post'] = tuple(synaptic_post)
    #
    #
    # def delete_delay_update(index):
    #     lastupdate =  list(synapse['lastupdate'])
    #     delay = list(synapse['delay'])
    #     lastupdate[0] = np.delete(lastupdate[0], index, axis=0)
    #     lastupdate[1] = lastupdate[1] - 1
    #     delay[0] = np.delete(delay[0], index, axis=0)
    #     delay[1] = delay[1] - 1
    #     synapse['lastupdate'] = tuple (lastupdate)
    #     synapse['delay'] = tuple(delay)


    net.store(name)
    state = net._stored_state[name]
    synapse = state[synapse_group.name]
    s_p = np.vstack((synapse['_synaptic_pre'][0],synapse['_synaptic_post'][0])).T
    s = np.hstack((np.array([source]), np.array([target])))
    try:
        index = np.where((s_p == s).all(1))[0][0]
        synapse['N'][0][0] = synapse['N'][0][0] -1
        synapse['N_outgoing'][0][source] = synapse['N_outgoing'][0][source] - 1
        synapse['N_incoming'][0][target] = synapse['N_incoming'][0][target] - 1

        s_c = synapse.copy()
        s_c.pop('N')
        s_c.pop('N_outgoing')
        s_c.pop('N_incoming')

        for c in s_c:
            delete_para(index,c)

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

P = PoissonGroup(5, np.arange(5)*Hz + 50*Hz)
G = NeuronGroup(5, equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms,name = 'neurongroup')
S = Synapses(P, G, 'w : 1',on_pre = on_pre, method='linear',name ='synapses')
S.connect(j =4, i= 4)
S.connect(j =3, i= 3)

S.delay = '0.1*j*ms'
S.w = '0.1+j*0.1'

m1 = StateMonitor(G,('v','I'),record=True)
m2 = SpikeMonitor(P)

net = Network(collect())
net.run(1*ms)
net.store('first')

add_group_neuron(net,G,2)
S.connect(i =2, j=6)
s_index=find_synapse(S,2,6)
S.w[s_index] = 0.1

visualise_connectivity(S)

add_neuron_mon(net,m1,G,6)

# G.equations._equations['I'] = "I = (g-h)*30 : 1"
# G.equations._equations.pop('I')
# G.equations = G.equations+("I = (g-h)*30 : 1")

net.run(100*ms)


fig1 = plt.figure(figsize=(10,4))
subplot(121)
plot(m1.t/ms,m1.v[5],'-b', label='Neuron 9 V')
legend()

subplot(122)
plot(m1.t/ms, m1.I[5], label='Neuron 9 I')
legend()

show()


#-----------Common used while debugging in Ipython---------------
# net.get_states()
# G.get_states()
#
# net._full_state()
# G._full_state()
#
# net.restore('agn')
# state = net._stored_state['agn']['synapses']
#
# ds=S._dpendencies
# dm=m1._dependencies
#
# S.add_dependency(G)
# m1.add_dependency(G)
#
# G.variables['N'].get_value()
# obj = net.objects

# m1.variables['v'].device.arrays[m1.variables['v']]._data.shape
# x = DynamicArray((2, 3), dtype=int)
