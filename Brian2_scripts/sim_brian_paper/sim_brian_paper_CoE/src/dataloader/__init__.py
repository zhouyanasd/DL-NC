# -*- coding: utf-8 -*-
"""
    The dataloader methods used for CoE.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from .KTH_dataloader import *
from .UCI_dataloader import *
from .BN_dataloader import *

import os, pickle

__all__ = [
    "KTH_classification",
    "UCI_classification",
    "BN_classification",
]

class Dataloader():

    def dump_data(self, path, dataset):
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'wb') as file:
            pickle.dump(dataset, file)

    def load_data(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)