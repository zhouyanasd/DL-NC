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
    def __init__(self, codes, config_SubCom, config_key):
        self.codes = codes
        self.config_SubCom = config_SubCom
        self.config_key = config_key

    def separator(self, Gen, target):
        key = self.config_key[target]
        SubCom = self.config_SubCom[target]
        parameter = {}
        for p,k in zip(Gen[SubCom],key):
            parameter[k] = p
        return parameter



if __name__ == "__main__":
    # keys = ['x', 'y', 'z', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    codes = np.array([None, None, None, None, None, 1, None, 1])
    # SubCom = np.array([[0, 1], [2, 3], [4, 5, 6, 7]])
    # Encoding = {}
    # Readout = {}
    Reservoir = ['N', 'L_1', 'L_2']
    Block_random = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P']
    Block_scale_free = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'C']
    Block_circle = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P_f', 'P_b', 'P_d', 'D_c']
    Block_hierarchy = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P_f', 'P_b', 'P_d', 'D_c', 'L']
    config_key = {'Reservoir':Reservoir, 'Block_random':Block_random, 'Block_scale_free':Block_scale_free,
              'Block_circle':Block_circle,'Block_hierarchy':Block_hierarchy}
    config_SubCom = {'Reservoir':[0, 1, 2], 'Block_random':[3, 4, 5, 6, 7, 8, 9],
                     'Block_scale_free': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                     'Block_circle': [20, 21, 22, 23,24,25,26,27,28,29,30],
                     'Block_hierarchy': [31,32,33,34,35,36,37,38,39,40,41,42]}

    decoder = Decoder(codes, config_SubCom, config_key)