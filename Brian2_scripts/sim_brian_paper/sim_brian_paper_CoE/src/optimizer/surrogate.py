from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.random_forest import RandomForestRegressor_
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.bayesian import UtilityFunction
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import re, warnings

import numpy as np


def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))

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


class Queue():
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


class Surrogate():
    def __init__(self, f, pbounds, random_state, model):
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # define used model e.g. gp or rf
        self.model = model

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
            while True:
                try:
                    x_probe = next(self._queue)
                except StopIteration:
                    self.update_model()
                    break
                self.probe(x_probe, lazy=False)

    def update_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self._space.params, self._space.target)

    def guess(self, X):
        gauss_value = self.model.guess_fixedpoint(X)
        return gauss_value

    def predict(self, X):
        predict_value = self.model.predict(X)
        return predict_value


class RandomForestRegressor_surrgate(Surrogate):
    def __init__(self, f, pbounds, random_state, n_Q, **rf_params):
        self._rf = RandomForestRegressor_(
            n_estimators=10,
            criterion="mse",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False
        )
        self._rf.set_params(**rf_params)

        self.n_Q = n_Q

        super(RandomForestRegressor_surrgate, self).__init__(
            f=f,
            pbounds=pbounds,
            random_state=random_state,
            model=self._rf
        )

    def guess(self, X):
        y_predict = self.model.predict(X)
        y_all_tree = np.array(self.model.y_hat_).T
        y_order_index = y_all_tree.argsort()
        y_selected_index = []
        y_selected = []
        for i in range(len(y_all_tree)):
            y_selected_index.append(self._uniform_select(y_order_index[i]))
            y_selected.append(y_all_tree[i][y_selected_index[i]])
        y_selected = np.array(y_selected).T
        y_predict = y_selected.mean(axis=0)
        return y_predict

    def predict(self, X):
        predict_value = self.guess(X)
        return predict_value

    def _uniform_select(self, index):
        n = len(index)
        idx = np.round(np.linspace(0, n - 1, self.n_Q)).astype(int)
        return index[idx]



class GaussianProcess_surrgate(Surrogate):
    def __init__(self, f, pbounds, random_state, acq='ucb', kappa=2.576, xi=0.0, **gp_params):
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=random_state,
        )
        self._gp.set_params(**gp_params)

        super(GaussianProcess_surrgate, self).__init__(
            f=f,
            pbounds=pbounds,
            random_state=random_state,
            model=self._gp
        )

        self.utility_function = UtilityFunction(kind=acq, kappa=kappa, xi=xi, bounds=self._space.bounds)

    def guess(self, X):
        y =self.utility_function.utility(X, self.model, self._space.target.min())
        return y


def create_surrogate(surrogate_type, f, pbounds, random_state, **surrogate_parameters):
    if surrogate_type == 'gp':
        acq = surrogate_parameters.pop('acq')
        kappa = surrogate_parameters.pop('kappa'),
        xi = surrogate_parameters.pop('xi'),
        surrogate = GaussianProcess_surrgate(f = f, pbounds = pbounds, random_state = random_state,
                                             acq = acq, kappa = kappa, xi = xi,
                                             **surrogate_parameters)
        return surrogate
    elif surrogate_type == 'rf':
        n_Q = surrogate_parameters.pop('n_Q')
        surrogate = RandomForestRegressor_surrgate(f = f, pbounds = pbounds, random_state = random_state,
                                                   n_Q = n_Q,
                                                   **surrogate_parameters)
        return surrogate
    else:
        raise ('Wrong surrogate type, only "gp" or "rl"')