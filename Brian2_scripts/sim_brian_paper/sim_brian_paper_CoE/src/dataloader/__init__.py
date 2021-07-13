# -*- coding: utf-8 -*-
"""
    The dataloader methods used for CoE.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from .KTH_dataloader import *
from .UCI_dataloader import *
from .BN_dataloader import *


__all__ = [
    "KTH_classification",
    "UCI_classification",
    "BN_classification",
]