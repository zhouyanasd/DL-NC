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
    def __init__(self, config_key, config_SubCom, config_codes, config_ranges, config_borders,
                      config_precisions, config_scales):
        self.config_key = config_key
        self.config_codes = config_codes
        self.config_SubCom = config_SubCom
        self.config_ranges = config_ranges
        self.config_borders = config_borders
        self.config_precisions = config_precisions
        self.config_scales = config_scales

    @property
    def get_keys(self):
        keys = []
        for group_name in self.config_key.keys():
            for key in self.config_key[group_name]:
                keys.append(group_name+'_'+key)
        return keys

    @property
    def get_SubCom(self):
        return list(self.config_SubCom.values())

    @property
    def get_dim(self):
        return len(self.get_keys)

    @property
    def get_codes(self):
        codes = []
        for group_codes in self.config_codes.values():
            codes.extend(group_codes)
        return np.array(codes)

    @property
    def get_ranges(self):
        ranges = []
        for group_ranges in self.config_ranges.values():
            ranges.extend(group_ranges)
        return np.array(ranges).T

    @property
    def get_borders(self):
        borders = []
        for group_borders in self.config_borders.values():
            borders.extend(group_borders)
        return np.array(borders).T

    @property
    def get_precisions(self):
        precisions = []
        for group_precisions in self.config_precisions.values():
            precisions.extend(group_precisions)
        return np.array(precisions)

    @property
    def get_scales(self):
        scales = []
        for group_scales in self.config_scales.values():
            scales.extend(group_scales)
        return np.array(scales)

    def register(self, Gen):
        self.Gen = Gen

    def decode(self, target):
        key = self.config_key[target]
        SubCom = self.config_SubCom[target]
        parameter = {}
        for p,k,c in zip(self.Gen[SubCom],key, self.codes):
            if c != None:
                p = self.dec2bin(p)
            parameter[k] = p
        return parameter



if __name__ == "__main__":
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
                     'Block_scale_free': [10, 11, 12, 13, 14, 15, 16],
                     'Block_circle': [17,18,19, 20, 21, 22, 23,24,25,26],
                     'Block_hierarchy': [27,28,29,30,31,32,33,34,35,36,37]}
    config_codes = {'Reservoir':[None, 1, 1], 'Block_random':[None, None, None, None, None, None, None],
                     'Block_scale_free': [None, None, None, None, None, None, None],
                     'Block_circle': [None, None, None, None, None, None, None, None, None, None, None],
                     'Block_hierarchy': [None, None, None, None, None, None, None, None, None, None, None, None]}
    config_ranges = {'Reservoir':[[0, 1]]*3, 'Block_random':[[0, 1]]*7,
                     'Block_scale_free': [[0, 1]]*7,
                     'Block_circle': [[0, 1]]*10,
                     'Block_hierarchy': [[0, 1]]*11}
    config_borders = {'Reservoir':[[0, 1]]*3, 'Block_random':[[0, 1]]*7,
                     'Block_scale_free': [[0, 1]]*7,
                     'Block_circle': [[0, 1]]*10,
                     'Block_hierarchy': [[0, 1]]*11}
    config_precisions = {'Reservoir':[4, 0, 0], 'Block_random':[4]*7,
                     'Block_scale_free': [4]*7,
                     'Block_circle': [4]*10,
                     'Block_hierarchy': [4]*11}
    config_scales = {'Reservoir':[0]*3, 'Block_random':[4]*7,
                     'Block_scale_free': [0]*7,
                     'Block_circle': [0]*10,
                     'Block_hierarchy': [0]*11}

    decoder = Decoder(config_key, config_SubCom, config_codes, config_ranges, config_borders,
                      config_precisions, config_scales)