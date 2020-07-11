# -*- coding: utf-8 -*-
"""
    The decoding method between genotype
    and the parameters.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions

class Decoder(BaseFunctions):
    def __init__(self, gen):
        self.gen = gen

    def decode(self):
        parameter = self.gen
        return parameter