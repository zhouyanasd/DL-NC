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

from Brian2_scripts.sim_brian_paper.sim_brian_paper_SNAS.src import *

from functools import partial
from multiprocessing import Pool

from brian2 import *
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)
data_path = '../../../Data/KTH/'

###################################
# -----simulation parameter setting-------
GenerateData = False
DataName = 'temp'

origin_size = (120, 160)
pool_size = (5, 5)
pool_types = 'max'
threshold = 0.2

F_train = 1
F_validation = 1
F_test = 1
Dt = defaultclock.dt = 1 * ms
standard_tau = 100

# -------class initialization----------------------
function = MathFunctions()
base = BaseFunctions()
readout = Readout()
KTH = KTH_classification()

# -------data initialization----------------------
try:
    df_en_train = KTH.load_data(data_path + 'train_' + DataName+'.p')
    df_en_validation = KTH.load_data(data_path + 'validation_' + DataName+'.p')
    df_en_test = KTH.load_data(data_path + 'test_' + DataName+'.p')

    data_train_s, label_train = KTH.get_series_data_list(df_en_train, is_group=True)
    data_validation_s, label_validation = KTH.get_series_data_list(df_en_validation, is_group=True)
    data_test_s, label_test = KTH.get_series_data_list(df_en_test, is_group=True)
except FileNotFoundError:
    GenerateData = True

# -------get numpy random state------------
np_state = np.random.get_state()


############################################
# ---- define network run function----
def run_net(inputs, **parameter):
    """
        run_net(inputs, parameter)
            Parameters = [R, p_inE/I, f_in, f_EE, f_EI, f_IE, f_II, tau_ex, tau_inh]
            ----------
    """

    # ---- set numpy random state for each run----
    np.random.set_state(np_state)

    # -----parameter setting-------
    n_ex = 1600
    n_inh = int(n_ex / 4)
    n_input = (origin_size[0] * origin_size[1]) / (pool_size[0] * pool_size[1])
    n_read = n_ex + n_inh

    R = parameter['R'] * 2
    f_in = parameter['f_in']
    f_EE = parameter['f_EE']
    f_EI = parameter['f_EI']
    f_IE = parameter['f_IE']
    f_II = parameter['f_II']

    A_EE = 60 * f_EE
    A_EI = 60 * f_EI
    A_IE = 60 * f_IE
    A_II = 60 * f_II
    A_inE = 60 * f_in
    A_inI = 60 * f_in

    tau_ex = parameter['tau_ex'] * standard_tau
    tau_inh = parameter['tau_inh'] * standard_tau
    tau_read = 30

    p_inE = parameter['p_in'] * 0.1
    p_inI = parameter['p_in'] * 0.1

    # ------definition of equation-------------
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
                        name='neurongroup_input')

    G_ex = NeuronGroup(n_ex, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                       name='neurongroup_ex')

    G_inh = NeuronGroup(n_inh, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                        name='neurongroup_in')

    G_readout = NeuronGroup(n_read, neuron_read, method='euler', name='neurongroup_read')

    S_inE = Synapses(Input, G_ex, synapse, on_pre=on_pre_ex, method='euler', name='synapses_inE')

    S_inI = Synapses(Input, G_inh, synapse, on_pre=on_pre_ex, method='euler', name='synapses_inI')

    S_EE = Synapses(G_ex, G_ex, synapse, on_pre=on_pre_ex, method='euler', name='synapses_EE')

    S_EI = Synapses(G_ex, G_inh, synapse, on_pre=on_pre_ex, method='euler', name='synapses_EI')

    S_IE = Synapses(G_inh, G_ex, synapse, on_pre=on_pre_inh, method='euler', name='synapses_IE')

    S_II = Synapses(G_inh, G_inh, synapse, on_pre=on_pre_inh, method='euler', name='synapses_I')

    S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_ex, method='euler')

    S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_inh, method='euler')

    # -------initialization of neuron parameters----------
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

    [G_ex, G_in] = base.allocate([G_ex, G_inh], 10, 10, 20)

    # -------initialization of network topology and synapses parameters----------
    S_inE.connect(condition='j<0.3*N_post', p=p_inE)
    S_inI.connect(condition='j<0.3*N_post', p=p_inI)
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
    duration = inputs[0].shape[0]
    net.run(duration * Dt)
    states = net.get_states()['neurongroup_read']['v']
    net.restore('init')
    return (states, inputs[1])


@Timelog
@AddParaName
def parameters_search(**parameter):
    # ------parallel run for train-------
    states_train_list = pool.map(partial(run_net, **parameter), [(x) for x in zip(data_train_s, label_train)])
    # ------parallel run for validation-------
    states_validation_list = pool.map(partial(run_net, **parameter),
                                      [(x) for x in zip(data_validation_s, label_validation)])
    # ----parallel run for test--------
    states_test_list = pool.map(partial(run_net, **parameter), [(x) for x in zip(data_test_s, label_test)])
    # ------Readout---------------
    states_train, states_validation, states_test, _label_train, _label_validation, _label_test = [], [], [], [], [], []
    for train in states_train_list:
        states_train.append(train[0])
        _label_train.append(train[1])
    for validation in states_validation_list:
        states_validation.append(validation[0])
        _label_validation.append(validation[1])
    for test in states_test_list:
        states_test.append(test[0])
        _label_test.append(test[1])
    states_train = (MinMaxScaler().fit_transform(np.asarray(states_train))).T
    states_validation = (MinMaxScaler().fit_transform(np.asarray(states_validation))).T
    states_test = (MinMaxScaler().fit_transform(np.asarray(states_test))).T
    score_train, score_validation, score_test = readout.readout_sk(states_train, states_validation, states_test,
                                                                   np.asarray(_label_train),
                                                                   np.asarray(_label_validation),
                                                                   np.asarray(_label_test), solver="lbfgs",
                                                                   multi_class="multinomial")
    # ----------show results-----------
    print('parameters %s' % parameter)
    print('Train score: ', score_train)
    print('Validation score: ', score_validation)
    print('Test score: ', score_test)
    return 1 - score_validation, 1 - score_test, 1 - score_train, parameter


##########################################
# -------optimizer settings---------------
if __name__ == '__main__':
    core = 8
    pool = Pool(core)
    parameters = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    bounds = {'R': (0.0001, 1), 'p_in': (0.0001, 1), 'f_in': (0.0001, 1), 'f_EE': (0.0001, 1), 'f_EI': (0.0001, 1),
              'f_IE': (0.0001, 1), 'f_II': (0.0001, 1), 'tau_ex': (0.0001, 1), 'tau_inh': (0.0001, 1)}
    parameters_search.func.keys = list(bounds.keys())

    LHS_path = './LHS_KTH.dat'
    SNAS = 'SAES'

    if GenerateData:
        KTH.load_data_KTH_all(data_path, split_type='mixed', split=[15, 5, 4])

        df_train = KTH.select_data_KTH(F_train, KTH.train, False)
        df_validation = KTH.select_data_KTH(F_validation, KTH.validation, False)
        df_test = KTH.select_data_KTH(F_train, KTH.test, False)

        df_en_train = KTH.encoding_latency_KTH(df_train, origin_size, pool_size, pool_types, threshold)
        df_en_validation = KTH.encoding_latency_KTH(df_validation, origin_size, pool_size, pool_types, threshold)
        df_en_test = KTH.encoding_latency_KTH(df_test, origin_size, pool_size, pool_types, threshold)

        KTH.dump_data(data_path + 'train_' + DataName, df_en_train)
        KTH.dump_data(data_path + 'validation_' + DataName, df_en_validation)
        KTH.dump_data(data_path + 'test_' + DataName, df_en_test)

    # -------parameters search---------------
    if SNAS == 'BO':
        optimizer = BayesianOptimization_(
            f=parameters_search,
            pbounds=bounds,
            random_state=np.random.RandomState(),
        )

        logger = bayes_opt.observer.JSONLogger(path="./BO_res_KTH.json")
        optimizer.subscribe(bayes_opt.event.Events.OPTMIZATION_STEP, logger)

        optimizer.minimize(
            LHS_path=LHS_path,
            init_points=50,
            is_LHS=True,
            n_iter=250,
            acq='ei',
            opt=optimizer.acq_min_DE,
        )

    elif SNAS == 'SAES':
        saes = SAES(parameters_search, 'ei', parameters, 0.5,
                    **{'ftarget': -1e+3, 'bounds': bounds, 'maxiter': 500,'tolstagnation': 500})
        saes.run_best_strategy(50, 1, 2, LHS_path=LHS_path)

    elif SNAS == 'CMA':
        res = cma.fmin(parameters_search, parameters, 0.5,
                       options={'ftarget': -1e+3, 'maxiter': 30,
                                'bounds': np.array([list(x) for x in list(bounds.values())]).T.tolist()})
