from brian2 import *
from multiprocessing import Pool
from cma import purecma


def aaa(inputs):
    N = 1
    print(inputs, id(N))
    return inputs+1

def parameters_search(parameter):
    # ---- check parameters >0 -----
    if (np.array(parameter)<0).any():
        return np.random.randint(10,100)

    states_train_list = pool.map(aaa, [1,2,3,4,5])
    print(states_train_list)
    return np.random.randint(5,100)


##########################################
# -------CMA-ES parameters search---------------
if __name__ == '__main__':
    core = 2
    pool = Pool(core)
    res = purecma.fmin(parameters_search, [30,1,1], 1, verb_disp=100)