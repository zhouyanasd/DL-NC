# -*- coding: utf-8 -*-
"""
    The decoding method between genotype
    and the parameters.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

class Decoder():
    def __init__(self, gen):
        self.gen = gen

    def decode(self):
        parameter = self.gen
        return parameter