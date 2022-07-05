# -*- coding: utf-8 -*-
"""
    The network runner function for different tasks.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from .task_KTH import *
from .task_HAPT import *
from .task_NMNIST import *

# --- tasks settings ---
tasks = {0: {'name':'HAPT', 'evaluator': task_HAPT_evaluator},
         1: {'name':'KTH', 'evaluator': task_KTH_evaluator},
         2: {'name':'NMNIST', 'evaluator': task_NMNIST_evaluator}}

__all__ = [
    "task_KTH_evaluator",
    "task_HAPT_evaluator",
    "task_NMNIST_evaluator",
]

__doc__ = """

"""

__author__ = 'Yan Zhou'
__version__ = "0.1.0  $Revision: 001 $ $Date: 2019-12-10 13:13:52 +0200 (Fri, 14 Jun 2019) $"