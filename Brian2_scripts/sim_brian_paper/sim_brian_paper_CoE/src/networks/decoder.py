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

    @property
    def get_SubCom(self):
        return list(self.config_SubCom.values())

    @property
    def get_keys(self):
        keys = []
        for group_name in self.config_key.keys():
            for key in self.config_key[group_name]:
                keys.append(group_name+'_'+key)
        return keys

    @property
    def get_dim(self):
        return len(self.get_keys)

    def get_codes(self):
        pass

    def get_ranges(self):
        pass

    def get_borders(self):
        pass

    def get_precisions(self):
        pass

    def get_scales(self):
        pass


    def register(self, Gen):
        self.Gen = Gen

    def separator(self, target):
        key = self.config_key[target]
        SubCom = self.config_SubCom[target]
        parameter = {}
        for p,k,c in zip(self.Gen[SubCom],key, self.codes):
            if c != None:
                p = self.dec2bin(p)
            parameter[k] = p
        return parameter



if __name__ == "__main__":
    codes = np.array([None, None, None, None, None, 1, None, 1])
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