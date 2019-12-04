from .core import BaseFunctions, MathFunctions, Timelog, AddParaName, Readout, Result
from .optimizer import BayesianOptimization_, SAES, cma
from .dataloader import MNIST_classification, KTH_classification

import bayes_opt

__all__ = [
    "BaseFunctions",
    "MathFunctions",
    "Timelog",
    "AddParaName",
    "Result",
    "Readout",
    "BayesianOptimization_",
    "SAES",
    "MNIST_classification",
    "KTH_classification",
    "cma",
    "bayes_opt"
]

__doc__ = """

"""

__author__ = 'Yan Zhou'
__version__ = "0.1.0  $Revision: 001 $ $Date: 2019-07-29 13:13:52 +0200 (Fri, 14 Jun 2019) $"
