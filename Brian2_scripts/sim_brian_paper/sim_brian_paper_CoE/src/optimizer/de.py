from numpy import asarray, zeros, zeros_like, tile, array, argmin, mod
from numpy.random import random, randint, rand, seed as rseed, uniform
import numpy as np

class _function_wrapper(object):
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)


class DiffEvol(object):
    def __init__(self, fun, bounds, npop, f=None, c=None, seed=None, maximize=False, vectorize=False, cbounds=(0.25, 1),
                 fbounds=(0.25, 0.75), pool=None, min_ptp=1e-2, args=[], kwargs={}):
        if seed is not None:
            rseed(seed)

        self.minfun = _function_wrapper(fun, args, kwargs)
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
