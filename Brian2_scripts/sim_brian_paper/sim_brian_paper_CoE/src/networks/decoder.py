# -*- coding: utf-8 -*-
"""
    The decoding method between genotype
    and the parameters.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions
import numpy as np
import geatpy as ga

class Decoder(BaseFunctions):
    def __init__(self, gen):
        self.gen = gen
        self.dim = 8
        self.keys = ['x', 'y', 'z', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
        self.ranges = np.vstack([[0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 9], [0, 10], [0, 9]]).T
        self.borders = np.vstack([[0, 0]] * self.dim).T
        self.precisions = np.array([4, 4, 4, 4, 4, 0, 4, 0])
        self.codes = np.array([None, None, None, None, None, 1, None, 1])
        self.scales = np.array([0] * self.dim)
        self.FieldDR = ga.crtfld(self.ranges, self.borders, self.precisions)
        self.SubCom = np.array([[0, 1], [2, 3], [4, 5, 6, 7]])

    def decode(self):
        parameter = self.gen
        return parameter