from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.surrogate import Surrogate

import threading

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.base import  _partition_estimators
from sklearn.utils.validation import check_is_fitted
from sklearn.externals.joblib import Parallel, delayed


def accumulate_prediction(predict, X, out, out_, lock):
    prediction = predict(X, check_input=False)
    out_.append(prediction)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


class RandomForestRegressor_(RandomForestRegressor):
    def __init__(self,
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
                 warm_start=False):
        super(RandomForestRegressor_, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        self.y_hat_ = []

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend="threading")(
            delayed(accumulate_prediction)(e.predict, X, [y_hat], self.y_hat_, lock)
            for e in self.estimators_)

        y_hat /= len(self.estimators_)

        return y_hat

class RandomForestRegressor_surrgate(Surrogate):
    def __init__(self, f, pbounds, random_state, **rf_params):

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

        super(RandomForestRegressor_surrgate, self).__init__(
            f = f,
            pbounds = pbounds,
            random_state=random_state,
            model = self._rf
        )