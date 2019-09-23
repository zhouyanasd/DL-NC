# -*- coding: utf-8 -*-
"""
    The optimization methods used for NAS.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""
import re
import warnings

import cma
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

import numpy as np
from scipy.stats import norm
from numpy import asarray, zeros, zeros_like, tile, array, argmin, mod
from numpy.random import random, randint, rand, seed as rseed, uniform


class DiffEvol(object):
    class _function_wrapper(object):
        def __init__(self, f, args, kwargs):
            self.f = f
            self.args = args
            self.kwargs = kwargs

        def __call__(self, x):
            return self.f(x, *self.args, **self.kwargs)

    def __init__(self, fun, bounds, npop, f=None, c=None, seed=None, maximize=False, vectorize=False, cbounds=(0.25, 1),
                 fbounds=(0.25, 0.75), pool=None, min_ptp=1e-2, args=[], kwargs={}):
        if seed is not None:
            rseed(seed)

        self.minfun = self._function_wrapper(fun, args, kwargs)
        self.bounds = asarray(bounds)
        self.n_pop = npop
        self.n_par = self.bounds.shape[0]
        self.bl = tile(self.bounds[:, 0], [npop, 1])
        self.bw = tile(self.bounds[:, 1] - self.bounds[:, 0], [npop, 1])
        self.m = -1 if maximize else 1
        self.pool = pool
        self.args = args

        if self.pool is not None:
            self.map = self.pool.map
        else:
            self.map = map

        self.periodic = []
        self.min_ptp = min_ptp

        self.cmin = cbounds[0]
        self.cmax = cbounds[1]
        self.cbounds = cbounds
        self.fbounds = fbounds

        self.seed = seed
        self.f = f
        self.c = c

        self._population = asarray(self.bl + random([self.n_pop, self.n_par]) * self.bw)
        self._fitness = zeros(npop)
        self._minidx = None

        self._trial_pop = zeros_like(self._population)
        self._trial_fit = zeros_like(self._fitness)

        if vectorize:
            self._eval = self._eval_vfun
        else:
            self._eval = self._eval_sfun

    @property
    def population(self):
        """The parameter vector population"""
        return self._population

    @property
    def minimum_value(self):
        """The best-fit value of the optimized function"""
        return self._fitness[self._minidx]

    @property
    def minimum_location(self):
        """The best-fit solution"""
        return self._population[self._minidx, :]

    @property
    def minimum_index(self):
        """Index of the best-fit solution"""
        return self._minidx

    def optimize(self, ngen):
        """Run the optimizer for ``ngen`` generations"""
        res = 0
        for res in self(ngen):
            pass
        return res

    def __call__(self, ngen=1):
        return self._eval(ngen)

    def evolve_population(self, pop, pop2, bound, f, c):
        npop, ndim = pop.shape

        for i in range(npop):

            # --- Vector selection ---
            v1, v2, v3 = i, i, i
            while v1 == i:
                v1 = randint(npop)
            while (v2 == i) or (v2 == v1):
                v2 = randint(npop)
            while (v3 == i) or (v3 == v2) or (v3 == v1):
                v3 = randint(npop)

            # --- Mutation ---
            v = pop[v1] + f * (pop[v2] - pop[v3])
            # random choice a value between when the solution out of the bounds
            for a, b in zip(enumerate(v), bound):
                if a[1] > b[1] or a[1] < b[0]:
                    v[a[0]] = np.random.uniform(b[0], b[1], 1)

            # --- Cross over ---
            co = rand(ndim)
            for j in range(ndim):
                if co[j] <= c:
                    pop2[i, j] = v[j]
                else:
                    pop2[i, j] = pop[i, j]

            # --- Forced crossing ---
            j = randint(ndim)
            pop2[i, j] = v[j]
        return pop2

    def _eval_sfun(self, ngen=1):
        """Run DE for a function that takes a single pv as an input and retuns a single value."""
        popc, fitc = self._population, self._fitness
        popt, fitt = self._trial_pop, self._trial_fit

        for ipop in range(self.n_pop):
            fitc[ipop] = self.m * self.minfun(popc[ipop, :])

        for igen in range(ngen):
            f = self.f or uniform(*self.fbounds)
            c = self.c or uniform(*self.cbounds)

            popt = self.evolve_population(popc, popt, self.bounds, f, c)
            fitt[:] = self.m * array(list(self.map(self.minfun, popt)))

            msk = fitt < fitc
            popc[msk, :] = popt[msk, :]
            fitc[msk] = fitt[msk]

            self._minidx = argmin(fitc)
            if fitc.ptp() < self.min_ptp:
                break

            yield popc[self._minidx, :], fitc[self._minidx]


class UtilityFunction_(UtilityFunction):
    def __init__(self, kind, kappa, xi):
        super(UtilityFunction_, self).__init__(kind, kappa, xi)

    def utility(self, x, gp, y_min):
        if self.kind == 'ucb':
            return self._ucb_(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei_(x, gp, y_min, self.xi)
        if self.kind == 'poi':
            return self._poi_(x, gp, y_min, self.xi)

    @staticmethod
    def _ucb_(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean - kappa * std

    @staticmethod
    def _ei_(x, gp, y_min, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (y_min - mean - xi) / std
        return -(y_min - mean - xi) * norm.cdf(z) - std * norm.pdf(z)

    @staticmethod
    def _poi_(x, gp, y_min, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (y_min - mean - xi) / std
        return -norm.cdf(z)


class BayesianOptimization_(BayesianOptimization):
    def __init__(self, f, pbounds, random_state=None, verbose=0):
        super(BayesianOptimization_, self).__init__(f, pbounds, random_state, verbose)

    def _prime_queue_LHS(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)
        LHS_points = self.LHSample(init_points, self._space.bounds)
        for point in LHS_points:
            self._queue.add(point)

    def LHSample(self, N, bounds, D=None):
        if D == None:
            D = bounds.shape[0]
        result = np.empty([N, D])
        temp = np.empty([N])
        d = 1.0 / N
        for i in range(D):
            for j in range(N):
                temp[j] = np.random.uniform(
                    low=j * d, high=(j + 1) * d, size=1)[0]
            np.random.shuffle(temp)
            for j in range(N):
                result[j, i] = temp[j]
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        if np.any(lower_bounds > upper_bounds):
            print('bounds error')
            return None
        np.add(np.multiply(result,
                           (upper_bounds - lower_bounds),
                           out=result),
               lower_bounds,
               out=result)
        return result

    def load_LHS(self, path):
        X, fit = [], []
        with open(path, 'r') as f:
            l = f.readlines()
        l.pop(0)
        p1 = re.compile(r'[{](.*?)[}]', re.S)
        for i in range(0, len(l)):
            l[i] = l[i].rstrip('\n')
            s = re.findall(p1, l[i])[0]
            d = eval('{' + s + '}')
            X.append(np.array(list(d.values())))
            f = float(l[i].replace('{' + s + '}', '').split(' ')[2])
            fit.append(f)
        return X, fit

    def acq_min_CMA(self, ac, gp, y_min, bounds, random_state):
        x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(bounds.shape[0]))
        options = {'tolfunhist': -1e+4, 'tolfun': -1e+4, 'ftarget': -1e+4, 'bounds': bounds.T.tolist(), 'maxiter': 1000,
                   'verb_log': 0, 'verb_time': False, 'verbose': -9}
        res = cma.fmin(lambda x: ac(x.reshape(1, -1), gp=gp, y_min=y_min), x_seeds, 0.25, options=options,
                       restarts=0, incpopsize=0, restart_from_best=False, bipop=False)
        x_min = res[0]
        return np.clip(x_min, bounds[:, 0], bounds[:, 1])

    def acq_min_DE(self, ac, gp, y_min, bounds, random_state, ngen=100, npop=45, f=0.4, c=0.3):
        de = DiffEvol(lambda x: ac(x.reshape(1, -1), gp=gp, y_min=y_min)[0], bounds, npop, f=f, c=c,
                      seed=random_state)
        de.optimize(ngen)
        print(de.minimum_value, de.minimum_location, de.minimum_index)
        x_min = de.minimum_location
        return np.clip(x_min, bounds[:, 0], bounds[:, 1])

    def suggest_(self, utility_function, opt_function):
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
        suggestion = opt_function(
            ac=utility_function.utility,
            gp=self._gp,
            y_min=self._space.target.min(),
            bounds=self._space.bounds,
            random_state=self._random_state.randint(100000)
        )
        return self._space.array_to_params(suggestion)

    def guess_fixedpoint(self, utility_function, X):
        gauss = utility_function.utility(X, self._gp, self._space.target.min())
        return gauss

    def minimize(self,
                 LHS_path=None,
                 init_points=5,
                 is_LHS=False,
                 n_iter=25,
                 acq='ucb',
                 opt=None,
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTMIZATION_START)
        if LHS_path == None:
            if is_LHS:
                self._prime_queue_LHS(init_points)
            else:
                self._prime_queue(init_points)
        else:
            X, fit = self.load_LHS(LHS_path)
            for x, eva in zip(X, fit):
                self.register(x, eva)
        if opt == None:
            opt = self.acq_min_DE
        self.set_gp_params(**gp_params)
        util = UtilityFunction_(kind=acq, kappa=kappa, xi=xi)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                x_probe = self.suggest_(util, opt)
                iteration += 1
            self.probe(x_probe, lazy=False)
        self.dispatch(Events.OPTMIZATION_END)


class SAES():
    def __init__(self, f, acquisition, x0, sigma, kappa=2.576, xi=0.0, **opts):
        self.f = f
        self.optimizer = BayesianOptimization_(
            f=f,
            pbounds=opts['bounds'],
            random_state=1,
        )
        self.util = UtilityFunction_(kind=acquisition, kappa=kappa, xi=xi)
        opts['bounds'] = self.optimizer._space._bounds.T.tolist()
        self.es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    def run_pre_selection(self, init_points, n, LHS_path=None):
        if LHS_path == None:
            LHS_points = self.optimizer.LHSample(np.clip(init_points - self.es.popsize, 1, np.inf).astype(int),
                                                 self.optimizer._space.bounds)  # LHS for BO
            fit_init = [self.f(**self.optimizer._space.array_to_params(x)) for x in
                        LHS_points]  # evaluated by the real fitness
            for x, eva in zip(LHS_points, fit_init):
                self.optimizer._space.register(x, eva)  # add LHS points to solution space
            X = self.es.ask()  # get the initial offstpring
            fit = [self.f(**self.optimizer._space.array_to_params(x)) for x in X]  # evaluated by the real fitness
            self.es.tell(X, fit)  # initial the CMA-ES model
            self.es.logger.add()  # update the log
            self.es.disp()
            for x, eva in zip(X, fit):
                self.optimizer._space.register(x, eva)  # update solution space
        else:
            X, fit = self.optimizer.load_LHS(LHS_path)
            for x, eva in zip(X, fit):
                self.optimizer._space.register(x, eva)  # add loaded LHS points to solution space
            self.es.ask()
            self.es.tell(X[-self.es.popsize:], fit[-self.es.popsize:])  # initial the CMA-ES model
            self.es.logger.add()  # update the log
            self.es.disp()
        self.optimizer._gp.fit(self.optimizer._space.params, self.optimizer._space.target)  # initialize the BO model
        while not self.es.stop():
            X = self.es.ask(self.es.popsize * n)  # initial n times offspring for pre-selection
            guess = self.optimizer.guess_fixedpoint(self.util, X)  # predice the possible good solution by BO
            X_ = np.array(X)[guess.argsort()[0:int(self.es.popsize)]]  # select the top n possible solution (minimum)
            fit_ = [self.f(**self.optimizer._space.array_to_params(x)) for x in X_]  # evaluted by real fitness function
            for x, eva in zip(X_, fit_):
                self.optimizer._space.register(x, eva)  # update solution space
            self.optimizer._gp.fit(self.optimizer._space.params,
                                   self.optimizer._space.target)  # update the BO model
            self.es.tell(X_, fit_)  # update the CMA-ES model
            self.es.logger.add()  # update the log
            self.es.disp()
        self.es.result_pretty()

    def run_best_strategy(self, init_points, n, inter=1, LHS_path=None):
        if LHS_path == None:
            LHS_points = self.optimizer.LHSample(np.clip(init_points - self.es.popsize, 1, np.inf).astype(int),
                                                 self.optimizer._space.bounds)  # LHS for BO
            fit_init = [self.f(**self.optimizer._space.array_to_params(x)) for x in
                        LHS_points]  # evaluated by the real fitness
            for x, eva in zip(LHS_points, fit_init):
                self.optimizer._space.register(x, eva)  # add LHS points to solution space
            X = self.es.ask()  # get the initial offstpring
            fit = [self.f(**self.optimizer._space.array_to_params(x)) for x in X]  # evaluated by the real fitness
            self.es.tell(X, fit)  # initial the CMA-ES model
            self.es.logger.add()  # update the log
            self.es.disp()
            for x, eva in zip(X, fit):
                self.optimizer._space.register(x, eva)  # update solution space
        else:
            X, fit = self.optimizer.load_LHS(LHS_path)
            for x, eva in zip(X, fit):
                self.optimizer._space.register(x, eva)  # add loaded LHS points to solution space
            self.es.ask()
            self.es.tell(X[-self.es.popsize:], fit[-self.es.popsize:])  # initial the CMA-ES model
            self.es.logger.add()  # update the log
            self.es.disp()
        self.optimizer._gp.fit(self.optimizer._space.params, self.optimizer._space.target)  # initialize the BO model
        estimation = 1  # counter
        while not self.es.stop():
            X = self.es.ask()  # initial offspring
            fit = self.optimizer._gp.predict(X)  # get the estimated value
            estimation += 1
            if estimation >= inter:
                estimation = 0  # initilize the counter
                guess = self.optimizer.guess_fixedpoint(self.util, X)  # predice the possible good solution by BO
                X_ = np.array(X)[guess.argsort()[0:int(n)]]  # select the top n possible solution (minimum)
                fit_ = [self.f(**self.optimizer._space.array_to_params(x)) for x in
                        X_]  # evaluted by real fitness function
                fit[guess.argsort()[0:int(n)]] = fit_  # replace the estimated value by real value
                for x, eva in zip(X_, fit_):
                    self.optimizer._space.register(x, eva)  # update the solution space
                self.optimizer._gp.fit(self.optimizer._space.params,
                                       self.optimizer._space.target)  # update the BO model
            self.es.tell(X, fit)  # update the CMA-ES model
            self.es.logger.add()  # update the log
            self.es.disp()
        self.es.result_pretty()
