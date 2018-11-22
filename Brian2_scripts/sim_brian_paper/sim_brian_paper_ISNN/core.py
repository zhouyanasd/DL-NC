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
        return int(index)
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

    net.store(name)
    delete_connected_synapses(net, neuron_group, index)
    net.restore(name)


def delete_group_synapse(net, synapse_group, source, target, name = 'dgn'):

    def delete_para(index,name):
        para =  list(synapse[name])
        para[0] = np.delete(para[0], index, axis=0)
        para[1] = para[1] - 1
        synapse[name] = tuple (para)

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
#--------------------------------------

# def a(a):
#     print(a)