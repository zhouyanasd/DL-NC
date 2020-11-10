from multiprocessing import Manager, Pool

from brian2 import *
start_scope()
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms

equ = '''
dv/dt = (I-v) / (20*ms) : 1 (unless refractory)
dg/dt = (-g)/(10*ms) : 1
dh/dt = (-h)/(9.5*ms) : 1
I = (g-h)*20 : 1
'''

on_pre ='''
h+=strength
g+=strength
'''

P = PoissonGroup(10, np.arange(10)*Hz + 50*Hz)
G = NeuronGroup(10, equ, threshold='v > 0.9', reset='v = 0', method='linear',refractory=1*ms )
S = Synapses(P, G, 'strength : 1',on_pre = on_pre, method='linear', delay = 1*ms, name='pathway_1')
S.connect(j='i')

S.strength = 1

net = Network(collect())
net.store('init')

def run_net(x, q):
    state = q.get()
    net._stored_state['temp'] = state[0]
    net.restore('temp')
    net.run(10 * ms)
    print(state[1])
    q.put([net._full_state(), state[1]])
    return x

def sum_strength(net, queue):
    net.restore('init')
    state_init = net._full_state()
    l = queue.qsize()
    states = []
    while not queue.empty():
        states.append(queue.get()[0])
    for com in list(state_init.keys()):
        if 'block_block_' in com or 'pathway_' in com and '_pre' not in com and '_post' not in com:
            try:
                np.subtract(state_init[com]['strength'][0], state_init[com]['strength'][0],
                       out=state_init[com]['strength'][0])
                for state in states:
                    np.add(state_init[com]['strength'][0], state[com]['strength'][0]/l,
                            out = state_init[com]['strength'][0])
                    print(com, state_init[com]['strength'][0])
            except:
                continue
    net._stored_state['init'] = state_init

if __name__ == '__main__':
    core = 4
    pool = Pool(core)
    q = Manager().Queue(core)
    for i in range(core):
        q.put([net._full_state(), i])

    result = pool.starmap(run_net, [(x, q) for x in range(20)])

    print('-----')
    sum_strength(net, q)
    print(net._stored_state['init']['pathway_1']['strength'])

