# -*- coding: utf-8 -*-
"""
    The components generator based on the parameter
    decoded from optimizer.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""


from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.networks import *

from brian2 import *


class Generator():
    def __init__(self, parameter):
        self.parameter = parameter

    def generate_block(self, N, ratio):
        block = Block(N, ratio)

    def generate_synapse(self):
        pass

    def generate_reservoir(self):
        pass