# -*- coding: utf-8 -*-
"""
    The Neural Structure Search (NAS) of large scale Liquid State Machine
    (LSM) for MNIST. The optimization method adopted
    here is CMA-ES, BO and Gaussian process assisted CMA-ES.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.

Requirement
=======
Numpy
Pandas
Brian2

Usage
=======

Citation
=======

"""

from .core import *
from .dataloader import *
from .optimizer import *

from functools import partial
from multiprocessing import Pool

warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)
data_path = '../../../Data/MNIST_data/'


###################################
# -----simulation parameter setting-------
coding_n = 3
MNIST_shape = (1, 784)
coding_duration = 30
duration = coding_duration*MNIST_shape[0]
F_train = 0.05
F_validation = 0.00833333
F_test = 0.05
Dt = defaultclock.dt = 1*ms

#-------class initialization----------------------
function = Function()
base = Base(duration, Dt)
readout = Readout(function.logistic)
result = Result()
MNIST = MNIST_classification(MNIST_shape, duration, Dt)

#-------data initialization----------------------
MNIST.load_Data_MNIST_all(data_path)
df_train_validation = MNIST.select_data(F_train+F_validation, MNIST.train)
df_train, df_validation = train_test_split(df_train_validation, test_size=F_validation/(F_validation+F_train),
                                           random_state=42)
df_test = MNIST.select_data(F_test, MNIST.test)

df_en_train = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_train, coding_n)
df_en_validation = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_validation, coding_n)
df_en_test = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_test, coding_n)

data_train_s, label_train = MNIST.get_series_data_list(df_en_train, is_group = True)
data_validation_s, label_validation = MNIST.get_series_data_list(df_en_validation, is_group = True)
data_test_s, label_test = MNIST.get_series_data_list(df_en_test, is_group = True)

#-------get numpy random state------------
np_state = np.random.get_state()


############################################
# ---- define network run function----
def run_net(inputs, parameter):
    """
        run_net(inputs, parameter)
            Parameters = [R, p_inE/I, f_in, f_EE, f_EI, f_IE, f_II, tau_ex, tau_inh]
            ----------
    """

    #---- set numpy random state for each run----
    np.random.set_state(np_state)

    # -----parameter setting-------
    n_ex = 1600
    n_inh = int(n_ex/4)
    n_input = MNIST_shape[1]*coding_n
    n_read = n_ex+n_inh

    R = parameter[0]
    f_in = parameter[2]
    f_EE = parameter[3]
    f_EI = parameter[4]
    f_IE = parameter[5]
    f_II = parameter[6]

    A_EE = 60*f_EE
    A_EI = 60*f_EI
    A_IE = 60*f_IE
    A_II = 60*f_II
    A_inE = 60*f_in
    A_inI = 60*f_in

    tau_ex = parameter[7]*coding_duration
    tau_inh = parameter[8]*coding_duration
    tau_read= 30

    p_inE = parameter[1]*0.1
    p_inI = parameter[1]*0.1

    #------definition of equation-------------
    neuron_in = '''
    I = stimulus(t,i) : 1
    '''

    neuron = '''
    tau : 1
    dv/dt = (I-v) / (tau*ms) : 1 (unless refractory)
    dg/dt = (-g)/(3*ms) : 1
    dh/dt = (-h)/(6*ms) : 1
    I = (g+h)+13.5: 1
    x : 1
    y : 1
    z : 1
    '''

    neuron_read = '''
    tau : 1
    dv/dt = (I-v) / (tau*ms) : 1
    dg/dt = (-g)/(3*ms) : 1 
    dh/dt = (-h)/(6*ms) : 1
    I = (g+h): 1
    '''

    synapse = '''
    w : 1
    '''

    on_pre_ex = '''
    g+=w
    '''

    on_pre_inh = '''
    h-=w
    '''

    # -----Neurons and Synapses setting-------
    Input = NeuronGroup(n_input, neuron_in, threshold='I > 0', method='euler', refractory=0 * ms,
                        name = 'neurongroup_input')

    G_ex = NeuronGroup(n_ex, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                    name ='neurongroup_ex')

    G_inh = NeuronGroup(n_inh, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                    name ='neurongroup_in')

    G_readout = NeuronGroup(n_read, neuron_read, method='euler', name='neurongroup_read')

    S_inE = Synapses(Input, G_ex, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_inE')

    S_inI = Synapses(Input, G_inh, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_inI')

    S_EE = Synapses(G_ex, G_ex, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_EE')

    S_EI = Synapses(G_ex, G_inh, synapse, on_pre = on_pre_ex ,method='euler', name='synapses_EI')

    S_IE = Synapses(G_inh, G_ex, synapse, on_pre = on_pre_inh ,method='euler', name='synapses_IE')

    S_II = Synapses(G_inh, G_inh, synapse, on_pre = on_pre_inh ,method='euler', name='synapses_I')

    S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_ex, method='euler')

    S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_inh, method='euler')

    #-------initialization of neuron parameters----------
    G_ex.v = '13.5+1.5*rand()'
    G_inh.v = '13.5+1.5*rand()'
    G_readout.v = '0'
    G_ex.g = '0'
    G_inh.g = '0'
    G_readout.g = '0'
    G_ex.h = '0'
    G_inh.h = '0'
    G_readout.h = '0'
    G_ex.tau = tau_ex
    G_inh.tau = tau_inh
    G_readout.tau = tau_read
    [G_ex,G_in] = base.allocate([G_ex,G_inh],10,10,20)


    # -------initialization of network topology and synapses parameters----------
    S_inE.connect(condition='j<0.3*N_post', p = p_inE)
    S_inI.connect(condition='j<0.3*N_post', p = p_inI)
    S_EE.connect(condition='i != j', p='0.3*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
    S_EI.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
    S_IE.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
    S_II.connect(condition='i != j', p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
    S_E_readout.connect(j='i')
    S_I_readout.connect(j='i+n_ex')

    S_inE.w = function.gamma(A_inE, S_inE.w.shape)
    S_inI.w = function.gamma(A_inI, S_inI.w.shape)
    S_EE.w = function.gamma(A_EE, S_EE.w.shape)
    S_IE.w = function.gamma(A_IE, S_IE.w.shape)
    S_EI.w = function.gamma(A_EI, S_EI.w.shape)
    S_II.w = function.gamma(A_II, S_II.w.shape)

    S_EE.pre.delay = '1.5*ms'
    S_EI.pre.delay = '0.8*ms'
    S_IE.pre.delay = '0.8*ms'
    S_II.pre.delay = '0.8*ms'

    # ------create network-------------
    net = Network(collect())
    net.store('init')

    # ------run network-------------
    stimulus = TimedArray(inputs[0], dt=Dt)
    net.run(duration * Dt)
    states = net.get_states()['neurongroup_read']['v']
    net.restore('init')
    return (states, inputs[1])

@Timelog
def parameters_search(parameter):
    # ------parallel run for train-------
    states_train_list = pool.map(partial(run_net, parameter = parameter), [(x) for x in zip(data_train_s, label_train)])
    # ------parallel run for validation-------
    states_validation_list = pool.map(partial(run_net, parameter = parameter), [(x) for x in zip(data_validation_s,
                                                                                                 label_validation)])
    # ----parallel run for test--------
    states_test_list = pool.map(partial(run_net, parameter = parameter), [(x) for x in zip(data_test_s, label_test)])
    # ------Readout---------------
    states_train, states_validation, states_test, _label_train, _label_validation, _label_test = [], [], [], [], [], []
    for train in states_train_list :
        states_train.append(train[0])
        _label_train.append(train[1])
    for validation in states_validation_list :
        states_validation.append(validation[0])
        _label_validation.append(validation[1])
    for test in states_test_list:
        states_test.append(test[0])
        _label_test.append(test[1])
    states_train = (MinMaxScaler().fit_transform(np.asarray(states_train))).T
    states_validation = (MinMaxScaler().fit_transform(np.asarray(states_validation))).T
    states_test = (MinMaxScaler().fit_transform(np.asarray(states_test))).T
    score_train, score_validation, score_test = readout.readout_sk(states_train, states_validation, states_test,
                                                 np.asarray(_label_train), np.asarray(_label_validation),
                                                 np.asarray(_label_test), solver="lbfgs", multi_class="multinomial")
    # ----------show results-----------
    print('parameters %s' % parameter)
    print('Train score: ', score_train)
    print('Validation score: ', score_validation)
    print('Test score: ', score_test)
    return 1-score_validation, 1 - score_test, 1 - score_train, parameter

##########################################
# -------CMA-ES parameters search---------------
if __name__ == '__main__':
    core = 10
    pool = Pool(core)
    parameters = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    res = cma.fmin(parameters_search, parameters, 0.5, options={'ftarget': -1e+3,'bounds': [0, 1],
                                                                 'maxiter':30})

