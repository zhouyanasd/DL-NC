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

Encoding = {}
Readout = {}
Reservoir = {'N','L_1','L_2'}
Block_random = ['N','tau','threshold','type','strength','plasticity','P']
Block_scale_free= {'N','tau','threshold','type','strength','plasticity','C'}
Block_circle = {'N','tau','threshold','type','strength','plasticity','P_f','P_b','P_d','D_c'}
Block_hierarchy = {'N','tau','threshold','type','strength','plasticity','P_f','P_b','P_d','D_c','L',''}

class Decoder(BaseFunctions):
    def __init__(self, **kwargs):
        self.keys = ['x', 'y', 'z', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
        self.codes = np.array([None, None, None, None, None, 1, None, 1])
        self.SubCom = np.array([[0, 1], [2, 3], [4, 5, 6, 7]])
        self.gen_group = kwargs

    def separator(self, gen):
        pass

    def gen_reader(self):
        pass

