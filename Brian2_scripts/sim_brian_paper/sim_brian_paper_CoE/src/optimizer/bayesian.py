from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.de import DiffEvol

import re, warnings

import numpy as np
from scipy.stats import norm, entropy
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import cma

def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class EntropySearch(object):
    """ Entropy search acquisition function

    This acquisition function samples at the position which reveals the maximal
    amount of information about the true position of the maximum. For this
    *n_candidates* data points (representers) for the position of the true
    maximum (p_max) are selected.
    From the GP model, *n_gp_samples* samples from the posterior are
    drawn and their entropy is computed. For each query point, the GP model is
    updated assuming *n_samples_y* outcomes (according to the current GP model).
    The change of entropy resulting from this assumed outcomes is computed and
    the query point which minimizes the entropy of p_max is selected.

    See also:
        Hennig, Philipp and Schuler, Christian J.
        Entropy Search for Information-Efficient Global Optimization.
        JMLR, 13:1809–1837, 2012.
    """
    def __init__(self, model, n_candidates=20, n_gp_samples=500,
                 n_samples_y=10, n_trial_points=500, rng_seed=0):
        self.model = model
        self.n_candidates = n_candidates
        self.n_gp_samples = n_gp_samples
        self.n_samples_y =  n_samples_y
        self.n_trial_points = n_trial_points
        self.rng_seed = rng_seed

        # We use an equidistant grid instead of sampling from the 1d normal
        # distribution over y
        equidistant_grid = np.linspace(0.0, 1.0, 2 * self.n_samples_y +1)[1::2]
        self.percent_points = norm.ppf(equidistant_grid)

    def __call__(self, x, incumbent=0, *args, **kwargs):
        """ Returns the change in entropy of p_max when sampling at x.

        Parameters
        ----------
        x: array-like
            The position(s) at which the upper confidence bound will be evaluated.
        incumbent: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning. Defaults to 0. [Not used by this acquisition
            function]

        Returns
        -------
        entropy_change: float
            the change in entropy of p_max when sampling at x.
        """
        x = np.atleast_2d(x)

        a_ES = np.empty((x.shape[0], self.n_samples_y))

        # Evaluate mean and covariance of GP at all representer points and
        # points x where MRS will be evaluated
        f_mean_all, f_cov_all = \
            self.model.gp.predict(np.vstack((self.X_candidate, x)),
                                  return_cov=True)
        f_mean = f_mean_all[:self.n_candidates]
        f_cov = f_cov_all[:self.n_candidates, :self.n_candidates]

        # Iterate over all x[i] at which we will evaluate the acquisition
        # function (often x.shape[0]=1)
        for i in range(self.n_candidates, self.n_candidates+x.shape[0]):
            # Simulate change of covariance (f_cov_delta) for a sample at x[i],
            # which actually would not depend on the observed value y[i]
            f_cov_query = f_cov_all[[i]][:, [i]]
            f_cov_cross = f_cov_all[:self.n_candidates, [i]]
            f_cov_query_inv = np.linalg.inv(f_cov_query)
            f_cov_delta = -np.dot(np.dot(f_cov_cross, f_cov_query_inv),
                                  f_cov_cross.T)

            # precompute samples from GP posterior for non-modified mean
            f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
                f_mean, f_cov + f_cov_delta, self.n_gp_samples).T

            # adapt for different outcomes y_i[j] of the query at x[i]
            for j in range(self.n_samples_y):
                # "sample" outcomes y_i[j] (more specifically where on the
                # Gaussian distribution over y_i[j] we would end up)
                y_delta = np.sqrt(f_cov_query + self.model.gp.alpha)[:, 0] \
                    * self.percent_points[j]
                # Compute change in GP mean at representer points
                f_mean_delta = f_cov_cross.dot(f_cov_query_inv).dot(y_delta)

                # Adapt samples to changes in GP posterior mean
                f_samples_j = f_samples + f_mean_delta[:, np.newaxis]
                # Count frequency of the candidates being the optima in the samples
                p_max = np.bincount(np.argmax(f_samples_j, 0),
                                    minlength=f_mean.shape[0]) \
                    / float(self.n_gp_samples)
                # Determing entropy of distr. p_max and compare to base entropy
                a_ES[i - self.n_candidates, j] = \
                    self.base_entropy - entropy(p_max)

         # Average entropy change over the different  assumed outcomes y_i[j]
        return a_ES.mean(1)

    def set_boundaries(self, boundaries, X_candidate=None):
        """Sets boundaries of search space.

        This method is assumed to be called once before running the
        optimization of the acquisition function.

        Parameters
        ----------
        boundaries: ndarray-like, shape=(n_params_dims, 2)
            Box constraint on search space. boundaries[:, 0] defines the lower
            bounds on the dimensions, boundaries[:, 1] defines the upper
            bounds.
        """
        self.X_candidate = X_candidate
        if self.X_candidate is None:
            # Sample n_candidates data points, which are checked for
            # being selected as representer points using (discretized) Thompson
            # sampling
            self.X_candidate = \
                np.empty((self.n_candidates, boundaries.shape[0]))
            for i in range(self.n_candidates):
                # SelectObjective n_trial_points data points uniform randomly
                candidates = np.random.uniform(
                    boundaries[:, 0], boundaries[:, 1],
                    (self.n_trial_points, boundaries.shape[0]))
                # Sample function from GP posterior and select the trial points
                # which maximizes the posterior sample as representer points
                try:
                    y_samples = self.model.gp.sample_y(candidates)
                    self.X_candidate[i] = candidates[np.argmax(y_samples)]
                except np.linalg.LinAlgError:  # This should happen very infrequently
                    self.X_candidate[i] = candidates[0]
        else:
            self.n_candidates = self.X_candidate.shape[0]

        ### Determine base entropy
        # Draw n_gp_samples functions from GP posterior
        f_mean, f_cov = \
            self.model.gp.predict(self.X_candidate, return_cov=True)
        f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
            f_mean, f_cov, self.n_gp_samples).T
        # Count frequency of the candidates being the optima in the samples
        p_max = np.bincount(np.argmax(f_samples, 0), minlength=f_mean.shape[0]) \
            / float(self.n_gp_samples)
        # Determing entropy of distr. p_max
        self.base_entropy = entropy(p_max)


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added

    Example
    -------
    # >>> def target_func(p1, p2):
    # >>>     return p1 + p2
    # >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    # >>> space = TargetSpace(target_func, pbounds, random_state=0)
    # >>> x = space.random_points(1)[0]
    # >>> y = space.register_point(x)
    # >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds, random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.

        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x): # in self
        return _hashable(x) in self._cache

    def __len__(self): # len(self)
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params, target):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        y : float
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in ammortized constant time

        Example
        -------
        # >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        # >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        # >>> len(space)
        # 0
        # >>> x = np.array([0, 0])
        # >>> y = 1
        # >>> space.add_observation(x, y)
        # >>> len(space)
        # 1
        """
        x = self._as_array(params)
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def probe(self, params):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Notes
        -----
        If x has been previously seen returns a cached value of y.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        y : float
            target function value.
        """
        x = self._as_array(params)

        try:
            target = self._cache[_hashable(x)]
        except KeyError:
            params = dict(zip(self._keys, x))
            target = self.target_func(**params)
            self.register(x, target)
        return target

    def random_sample(self):
        """
        Creates random points within the bounds of the space.

        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Example
        -------
        # >>> target_func = lambda p1, p2: p1 + p2
        # >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        # >>> space = TargetSpace(target_func, pbounds, random_state=0)
        # >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()

    def max(self):
        """Get maximum target value found and corresponding parametes."""
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        """Get all target values found and corresponding parametes."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class UtilityFunction():
    def __init__(self, kind, kappa, xi, bounds):
        self.kappa = kappa

        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi','es']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

        self.bounds = bounds

    def utility(self, x, gp, y_min):
        if self.kind == 'ucb':
            return self._ucb_(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei_(x, gp, y_min, self.xi)
        if self.kind == 'poi':
            return self._poi_(x, gp, y_min, self.xi)
        if self.kink == 'es':
            entropy_search = EntropySearch(gp)
            entropy_search.set_boundaries(self.bounds)
            return entropy_search(x, y_min)

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


class BayesianOptimization():
    def __init__(self, f, pbounds, random_state=None, acq='ucb', opt='de', kappa=2.576, xi=0.0,
                  verbose=0, **gp_params):

        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=self._random_state,
        )
        self._gp.set_params(**gp_params)

        if opt == 'cma':
            self.opt_function = self.acq_min_CMA
        else:
            self.opt_function = self.acq_min_DE

        self.utility_function = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        self._verbose = verbose

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            return self._space.probe(params)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)
        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

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

    def update_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

    def suggest(self):
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())
        suggestion = self.opt_function(
            ac=self.utility_function.utility,
            gp=self._gp,
            y_min=self._space.target.min(),
            bounds=self._space.bounds,
            random_state=self._random_state.randint(100000)
        )
        return self._space.array_to_params(suggestion)

    def guess_fixedpoint(self, X):
        gauss =self.utility_function.utility(X, self._gp, self._space.target.min())
        return gauss

    def initial_model(self,
                     LHS_path=None,
                     init_points=5,
                     is_LHS=False,
                     lazy = True,
                     ):
        if LHS_path == None:
            if is_LHS:
                self._prime_queue_LHS(init_points)
            else:
                self._prime_queue(init_points)
        else:
            X, fit = self.load_LHS(LHS_path)
            for x, eva in zip(X, fit):
                self.register(x, eva)
        if not lazy:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                self.update_model()
            self.probe(x_probe, lazy=False)

    def minimize(self,
                 LHS_path=None,
                 init_points=5,
                 is_LHS=False,
                 n_iter=25,
                 ):
        """Mazimize your function"""

        self.initial_model(LHS_path, init_points, is_LHS)

        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                self.update_model()
                x_probe = self.suggest()
                iteration += 1
            self.probe(x_probe, lazy=False)