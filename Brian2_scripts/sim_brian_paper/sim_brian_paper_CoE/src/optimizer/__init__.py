# -*- coding: utf-8 -*-
"""
    The optimization methods used for NAS.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from .bayesian import *
from .coe import *
from .de import *
from .surrogate import *

__all__ = [
    "DiffEvol",
    "BayesianOptimization",
    "RandomForestRegressor_surrogate",
    "RandomForestRegressor_surrogate_wang",
    "GaussianProcess_surrogate",
    "CoE",
    "CoE_surrogate",
]