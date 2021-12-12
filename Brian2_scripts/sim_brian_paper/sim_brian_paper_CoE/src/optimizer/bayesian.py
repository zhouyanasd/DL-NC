from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.de import DiffEvol
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.surrogate import Surrogate
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.utility_functions import UtilityFunction
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.random_forest import RandomForestRegressor

import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import cma

class BayesianOptimization(Surrogate):
    def __init__(self, f, keys, ranges, borders, precisions, model, random_state=None,
                 acq='lcb', opt='de', kappa=2.576, xi=0.0):
        super(BayesianOptimization, self).__init__(
            f = f,
            keys = keys,
            ranges = ranges,
            borders = borders,
            precisions = precisions,
            random_state=random_state,
            model = model
        )

        if opt == 'cma':
            self.opt_function = self.acq_min_CMA
        else:
            self.opt_function = self.acq_min_DE

        self.utility_function = UtilityFunction(kind=acq, kappa=kappa, xi=xi, bounds = self._space.bounds)

    def acq_min_CMA(self, ac, model, y_min, bounds, random_state):
        x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(bounds.shape[0]))
        options = {'tolfunhist': -1e+4, 'tolfun': -1e+4, 'ftarget': -1e+4, 'bounds': bounds.T.tolist(), 'maxiter': 1000,
                   'verb_log': 0, 'verb_time': False, 'verbose': -9}
        res = cma.fmin(lambda x: ac(x.reshape(1, -1), model=model, y_min=y_min), x_seeds, 0.25, options=options,
                       restarts=0, incpopsize=0, restart_from_best=False, bipop=False)
        x_min = res[0]
        return np.clip(x_min, bounds[:, 0], bounds[:, 1])

    def acq_min_DE(self, ac, model, y_min, bounds, random_state, ngen=100, npop=45, f=0.4, c=0.3):
        de = DiffEvol(lambda x: ac(x.reshape(1, -1), model=model, y_min=y_min)[0], bounds, npop, f=f, c=c,
                      seed=random_state)
        de.optimize(ngen)
        x_min = de.minimum_location
        return np.clip(x_min, bounds[:, 0], bounds[:, 1])

    def suggest(self):
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())
        suggestion = self.opt_function(
            ac=self.utility_function.utility,
            model=self.model,
            y_min=self._space.target.min(),
            bounds=self._space.bounds,
            random_state=self._random_state.randint(100000)
        )
        return self._space.array_to_params(self._space.add_precision(suggestion,self._space.precisions))

    def minimize(self,
                 LHS_path=None,
                 init_points=5,
                 is_LHS=False,
                 n_iter=25,
                 online = True,
                 show = False,
                 ):
        """Mazimize your function"""

        self.initial_model(LHS_path, init_points, is_LHS)

        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                if online:
                    self.update_model()
                x_probe = self.suggest()
                iteration += 1
            self.probe(x_probe, lazy=False)
            if show:
                print(self._space.min())

class GaussianProcess_BayesianOptimization(BayesianOptimization):
    def __init__(self, f, keys, ranges, borders, precisions, random_state,
                 acq='lcb', opt='de', kappa=2.576, xi=0.0, **gp_params):
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=random_state,
        )
        self._gp.set_params(**gp_params)

        super(GaussianProcess_BayesianOptimization, self).__init__(
            f=f,
            keys = keys,
            ranges = ranges,
            borders = borders,
            precisions = precisions,
            random_state=random_state,
            acq=acq,
            opt=opt,
            kappa=kappa,
            xi=xi,
            model=self._gp
        )


class RandomForestRegressor_BayesianOptimization(BayesianOptimization):
    def __init__(self, f, keys, ranges, borders, precisions, random_state,
                 acq='lcb', opt='de', kappa=2.576, xi=0.0, **rf_params):

        if acq == 'es':
            print('warning: Entropy search acquisition function should use gp model')

        self._rf = RandomForestRegressor(
            n_estimators=10,
            criterion="mse",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            bootstrap=True,
            oob_score=False,
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False,
            min_variance=0.0,
        )
        self._rf.set_params(**rf_params)

        super(RandomForestRegressor_BayesianOptimization, self).__init__(
            f=f,
            keys = keys,
            ranges = ranges,
            borders = borders,
            precisions = precisions,
            random_state=random_state,
            acq=acq,
            opt=opt,
            kappa=kappa,
            xi=xi,
            model=self._rf
        )