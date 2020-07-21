# -*- coding: utf-8 -*-
"""
    The optimization methods used for NAS.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from .bayesian import *
from .coe import *
from .de import *

__all__ = [
    "DiffEvol",
    "BayesianOptimization",
    "CoE_surrogate",
    "Coe_surrogate_mixgentype",
]