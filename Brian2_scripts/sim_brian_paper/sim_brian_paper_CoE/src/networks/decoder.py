# -*- coding: utf-8 -*-
"""
    The decoding method between genotype
    and the parameters.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions
import numpy as np

class Decoder(BaseFunctions):
    def __init__(self, config_group, config_keys, config_SubCom, config_codes, config_ranges, config_borders,
                      config_precisions, config_scales):
        super(Decoder).__init__()
        self.config_group = config_group
        self.config_keys = config_keys
        self.config_codes = config_codes
        self.config_SubCom = config_SubCom
        self.config_ranges = config_ranges
        self.config_borders = config_borders
        self.config_precisions = config_precisions
        self.config_scales = config_scales
        self.block_decoder_type = {'random':self.decode_block_random,
                                   'scale_free':self.decode_block_scale_free,
                                   'circle':self.decode_block_circle,
                                   'hierarchy':self.decode_block_hierarchy}

    @property
    def get_keys(self):
        keys = []
        for index, group_name in enumerate(self.config_group):
            for key in self.config_keys[index]:
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
        key = self.config_keys[group]
        SubCom = self.config_SubCom[group]
        codes = self.config_codes[group]
        parameter = {}
        for p,k,c in zip(self.Gen[SubCom], key, codes):
            if c != None:
                p = self.dec2bin(p)
            parameter[k] = p
        return parameter

    def decode_block_random(self):
        parameter = self.decode('Block_random')
        return parameter

    def decode_block_scale_free(self):
        parameter = self.decode('Block_scale_free')
        return parameter

    def decode_block_circle(self):
        parameter = self.decode('Block_circle')
        return parameter

    def decode_block_hierarchy(self):
        parameter = self.decode('Block_hierarchy')
        return parameter

    def get_reservoir_block_type(self):
        parameter = self.decode('Reservoir')
        return parameter['block']

    def get_reservoir_structure_type(self):
        parameter = self.decode('Reservoir')
        return parameter['layer_1', 'layer_2'].values()

    def get_encoding_structure(self):
        pass

    def get_readout_structure(self):
        pass

    def get_parameters_blocks(self):
        pass

    def get_parameters_reservoir(self):
        pass

    def get_parameters_encoding(self):
        pass

    def get_parameters_readout(self):
        pass

    def get_parameters_network(self):
        pass