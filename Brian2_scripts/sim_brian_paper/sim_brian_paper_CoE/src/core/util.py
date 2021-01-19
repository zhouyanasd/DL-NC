# -*- coding: utf-8 -*-
"""
    The basic function defined for the simulation.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt


class Timelog():
    def __init__(self, func):
        self.func = func
        self.itime = time.time()
        self.iteration = 0
        self.load_continue = False

    def __call__(self, *args, **kwargs):
        validation, test, train, parameters = self.func(*args, **kwargs)
        if self.load_continue:
            self.load_continue = False
            self.iteration, time = self.load()
            self.itime = self.itime - time
        self.save(validation, test, train, parameters)
        return validation

    @property
    def elapsed(self):
        return time.time() - self.itime

    def load(self):
        with open('Results_Record' + '.dat', 'r') as f:
            l = f.readlines()
        l.pop(0)
        return  int(l[-1].split(' ')[0]), float(l[-1].split(' ')[1])

    def save(self, validation, test, train, parameters):
        self.iteration += 1
        if self.iteration == 1:
            with open('Results_Record' + '.dat', 'w') as f:
                f.write('iteration' + ' '
                        + 'wall_time' + ' '
                        + 'result_validation' + ' '
                        + 'result_test' + ' '
                        + 'result_train' + ' '
                        + 'parameters' + ' '
                        + '\n')
        with open('Results_Record' + '.dat', 'a') as f:
            f.write(str(self.iteration) + ' ' + str(self.elapsed) + ' ' + str(validation) + ' '
                    + str(test) + ' ' + str(train) + ' ' + str(parameters) + ' ' + '\n')


class ProgressBar():
    """
    Example
    -------
    @ProgressBar
    def aa(**kwargs):
        time.sleep(0.05)

    n = aa.total = 100

    for i in range(n):
        aa()

    """
    def __init__(self, func):
        self.func = func
        self.total = 0
        self.now = 0

    def __call__(self, *arg, **kwargs):
        self.now += 1
        print("\r", end="")
        print("Runing progress: {:^3.0f}%: ".format((self.now)/self.total*100), "▋" * (int(self.now/self.total*100) // 2), end="\n")
        return self.func(**kwargs)



class Result():
    """
    Some functions for dealing with the results.
    """

    def __init__(self):
        pass

    def result_save(self, path, **kwarg):
        if os.path.exists(path):
            os.remove(path)
        fw = open(path, 'wb')
        pickle.dump(kwarg, fw)
        fw.close()

    def result_pick(self, path):
        fr = open(path, 'rb')
        data = pickle.load(fr)
        fr.close()
        return data

    def show_weights(self, f_init, gen):
        net = f_init(gen)
        net.restore('pre_run', 'pre_run_state.txt')
        coms = net.get_states()
        all_strength = []
        for com in list(coms.keys()):
            if 'block_block_' in com or 'pathway_' in com and '_pre' not in com and '_post' not in com:
                try:
                    all_strength.extend(list(net.get_states()[com]['strength']))
                    # print(com)
                except:
                    continue
        fig_distribution_w_EE = plt.figure(figsize=(5, 5))
        plt.hist(np.array(all_strength), 100)
        plt.xlabel('Weight')
        plt.show()