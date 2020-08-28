from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.de import DiffEvol
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.surrogate import Surrogate

import warnings

import numpy as np
from scipy.stats import norm, entropy
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import cma

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
            self.model.predict(np.vstack((self.X_candidate, x)),
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
                y_delta = np.sqrt(f_cov_query + self.model.alpha)[:, 0] \
                    * self.percent_points[j]
                # Compute change in GP mean at representer points
                f_mean_delta = f_cov_cross.dot(f_cov_query_inv).dot(y_delta)

                # Adapt samples to changes in GP posterior mean
                f_samples_j = f_samples + f_mean_delta[:, np.newaxis]
                # Count frequency of the candidates being the optima in the samples
                p_min = np.bincount(np.argmin(f_samples_j, 0),
                                    minlength=f_mean.shape[0]) \
                    / float(self.n_gp_samples)
                # Determing entropy of distr. p_max and compare to base entropy
                a_ES[i - self.n_candidates, j] = \
                    - self.base_entropy + entropy(p_min)

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
                    y_samples = self.model.sample_y(candidates)
                    self.X_candidate[i] = candidates[np.argmin(y_samples)]
                except np.linalg.LinAlgError:  # This should happen very infrequently
                    self.X_candidate[i] = candidates[0]
        else:
            self.n_candidates = self.X_candidate.shape[0]

        ### Determine base entropy
        # Draw n_gp_samples functions from GP posterior
        f_mean, f_cov = \
            self.model.predict(self.X_candidate, return_cov=True)
        f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
            f_mean, f_cov, self.n_gp_samples).T
        # Count frequency of the candidates being the optima in the samples
        p_min = np.bincount(np.argmin(f_samples, 0), minlength=f_mean.shape[0]) \
            / float(self.n_gp_samples)
        # Determing entropy of distr. p_max
        self.base_entropy = entropy(p_min)


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
        if self.kind == 'es':
            entropy_search = EntropySearch(gp, n_candidates=20, n_gp_samples=500,
                 n_samples_y=10, n_trial_points=500, rng_seed=0)
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


class BayesianOptimization(Surrogate):
    def __init__(self, f, pbounds, random_state=None, acq='ucb', opt='de', kappa=2.576, xi=0.0,
                  verbose=0, **gp_params):
        super(BayesianOptimization, self).__init__(
            f = f,
            pbounds = pbounds,
            random_state=random_state
        )

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

        self.utility_function = UtilityFunction(kind=acq, kappa=kappa, xi=xi, bounds = self._space.bounds)

        self._verbose = verbose

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