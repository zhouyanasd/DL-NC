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
    def __init__(self, config_group, config_key, config_SubCom, config_codes, config_ranges, config_borders,
                      config_precisions, config_scales):
        super.__init__()
        self.config_group = config_group
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
        for index, group_name in enumerate(self.config_group):
            for key in self.config_key[index]:
                keys.append(group_name+'_'+key)
        return keys

    @property
    def get_SubCom(self):
        return self.config_SubCom

    @property
    def get_dim(self):
        return len(self.get_keys)

    @property
    def get_codes(self):
        codes = []
        for group_codes in self.config_codes:
            codes.extend(group_codes)
        return np.array(codes)

    @property
    def get_ranges(self):
        ranges = []
        for group_ranges in self.config_ranges:
            ranges.extend(group_ranges)
        return np.array(ranges).T

    @property
    def get_borders(self):
        borders = []
        for group_borders in self.config_borders:
            borders.extend(group_borders)
        return np.array(borders).T

    @property
    def get_precisions(self):
        precisions = []
        for group_precisions in self.config_precisions:
            precisions.extend(group_precisions)
        return np.array(precisions)

    @property
    def get_scales(self):
        scales = []
        for group_scales in self.config_scales:
            scales.extend(group_scales)
        return np.array(scales)

    def register(self, Gen):
        self.Gen = Gen

    def decode(self, target):
        group = np.where(self.config_group == target)[0]
        key = self.config_key[group]
        SubCom = self.config_SubCom[group]
        codes = self.config_codes[group]
        parameter = {}
        for p,k,c in zip(self.Gen[SubCom], key, codes):
            if c != None:
                p = self.dec2bin(p)
            parameter[k] = p
        return parameter



if __name__ == "__main__":
    Encoding = ['N']
    Readout = ['N']
    Reservoir = ['N', 'L_1', 'L_2']
    Block_random = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P']
    Block_scale_free = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'C']
    Block_circle = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P_f', 'P_b', 'P_d', 'D_c']
    Block_hierarchy = ['N', 'tau', 'threshold', 'type', 'strength', 'plasticity', 'P_f', 'P_b', 'P_d', 'D_c', 'L']

    config_group = ['Encoding', 'Reservoir', 'Block_random', 'Block_scale_free',
                    'Block_circle','Block_hierarchy', 'Readout']

    config_key = {Encoding, Reservoir, Block_random, Block_scale_free,
                  Block_circle, Block_hierarchy, Readout}
    config_SubCom = [[38],[0, 1, 2], [3, 4, 5, 6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15, 16],
                     [17,18,19, 20, 21, 22, 23,24,25,26],
                     [27,28,29,30,31,32,33,34,35,36,37],[39]]
    config_codes = [[None],[None, 1, 1], [None, None, None, None, None, None, None],
                     [None, None, None, None, None, None, None],
                     [None, None, None, None, None, None, None, None, None, None, None],
                     [None, None, None, None, None, None, None, None, None, None, None, None],[None]]
    config_ranges = [[[0,1]],[[0, 1]]*3, [[0, 1]]*7,
                     [[0, 1]]*7,
                     [[0, 1]]*10,
                     [[0, 1]]*11,[[0,1]]]
    config_borders = [[[0,1]],[[0, 1]]*3, [[0, 1]]*7,
                     [[0, 1]]*7,
                     [[0, 1]]*10,
                     [[0, 1]]*11,[[0,1]]]
    config_precisions = [[0], [4, 0, 0], [4]*7,
                     [4]*7,
                     [4]*10,
                     [4]*11,[0]]
    config_scales = [[0],[0]*3, [0]*7,
                     [0]*7,
                     [0]*10,
                     [0]*11,[0]]

    decoder = Decoder(config_group, config_key, config_SubCom, config_codes, config_ranges, config_borders,
                      config_precisions, config_scales)