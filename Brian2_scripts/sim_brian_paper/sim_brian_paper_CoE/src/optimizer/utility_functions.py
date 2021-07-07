import numpy as np
from scipy.stats import norm, entropy

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

    def utility(self, x, model, y_min):
        if self.kind == 'ucb':
            return self._ucb_(x, model, self.kappa)
        if self.kind == 'ei':
            return self._ei_(x, model, y_min, self.xi)
        if self.kind == 'poi':
            return self._poi_(x, model, y_min, self.xi)
        if self.kind == 'es':
            entropy_search = EntropySearch(model, n_candidates=20, n_gp_samples=500,
                 n_samples_y=10, n_trial_points=500, rng_seed=0)
            entropy_search.set_boundaries(self.bounds)
            return entropy_search(x, y_min)

    @staticmethod
    def _ucb_(x, model, kappa):
        mean, std = model.predict(x, return_std=True)
        return mean - kappa * std

    @staticmethod
    def _ei_(x, model, y_min, xi):
        mean, std = model.predict(x, return_std=True)
        z = (y_min - mean - xi) / std
        return -(y_min - mean - xi) * norm.cdf(z) - std * norm.pdf(z)

    @staticmethod
    def _poi_(x, model, y_min, xi):
        mean, std = model.predict(x, return_std=True)
        z = (y_min - mean - xi) / std
        return -norm.cdf(z)