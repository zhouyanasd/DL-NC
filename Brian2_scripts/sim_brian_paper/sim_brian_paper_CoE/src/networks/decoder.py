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

    def set_structure_settings(self, **structure):
        self.structure_settings = structure

    def register(self, Gen):
        self.Gen = Gen

    def decode(self, target):
        group = np.where(np.array(self.config_group) == target)[0][0]
        key = self.config_keys[group]
        SubCom = self.config_SubCom[group]
        codes = self.config_codes[group]
        ranges = self.config_ranges[group]
        parameter = {}
        for p,k,c,r in zip(np.array(self.Gen)[SubCom], key, codes, ranges):
            if c != None:
                l = len(self.dec2bin((r[1]-r[0]), 0))
                p = self.dec2bin(p, l)
            parameter[k] = p
        return parameter

    def decode_block_random(self, need):
        parameters = self.decode('Block_random')
        if need == 'structure':
            sub_parameters = self.get_sub_dict(parameters, 'N', 'p')
            return sub_parameters
        if need == 'parameter':
            sub_parameters = self.get_sub_dict(parameters, 'plasticity', 'strength', 'tau', 'threshold', 'type')
            return sub_parameters

    def decode_block_scale_free(self, need):
        parameters = self.decode('Block_scale_free')
        if need == 'structure':
            sub_parameters = self.get_sub_dict(parameters, 'N', 'p_alpha', 'p_beta', 'p_gama')
            return sub_parameters
        if need == 'parameter':
            sub_parameters = self.get_sub_dict(parameters, 'plasticity', 'strength', 'tau', 'threshold', 'type')
            return sub_parameters

    def decode_block_circle(self, need):
        parameters = self.decode('Block_circle')
        if need == 'structure':
            sub_parameters = self.get_sub_dict(parameters, 'N', 'p_backward', 'p_forward', 'p_threshold')
            return sub_parameters
        if need == 'parameter':
            sub_parameters = self.get_sub_dict(parameters, 'plasticity', 'strength', 'tau', 'threshold', 'type')
            return sub_parameters

    def decode_block_hierarchy(self, need):
        parameters = self.decode('Block_hierarchy')
        if need == 'structure':
            sub_parameters = self.get_sub_dict(parameters, 'N_h', 'N_i', 'N_o', 'decay', 'p_in', 'p_out')
            return sub_parameters
        if need == 'parameter':
            sub_parameters = self.get_sub_dict(parameters, 'plasticity', 'strength', 'tau', 'threshold', 'type')
            return sub_parameters

    def get_reservoir_block_type(self):
        parameter = self.decode('Reservoir')
        type_b = parameter['block']
        type_d = [self.bin2dec(type_b[0:2]), self.bin2dec(type_b[2:4]),
                  self.bin2dec(type_b[4:6]), self.bin2dec(type_b[6:])]
        type_s = [self.structure_settings['structure_blocks']['components_' + str(type_d[0])],
                  self.structure_settings['structure_blocks']['components_' + str(type_d[1])],
                  self.structure_settings['structure_blocks']['components_' + str(type_d[2])],
                  self.structure_settings['structure_blocks']['components_' + str(type_d[3])]]
        return type_s

    def get_reservoir_structure_type(self):
        parameter = self.decode('Reservoir')
        type_b = parameter['layer_1'], parameter['layer_2']
        type_d = [self.bin2dec(type_b[0][0:2]), self.bin2dec(type_b[0][2:4]),
                  self.bin2dec(type_b[0][4:6]), self.bin2dec(type_b[0][6:])],\
                 [self.bin2dec(type_b[1][0:2]), self.bin2dec(type_b[1][2:4]),
                  self.bin2dec(type_b[1][4:6]), self.bin2dec(type_b[1][6:])]
        type_s = [self.structure_settings['structure_layer']['components_' + str(type_d[0][0])],
                  self.structure_settings['structure_layer']['components_' + str(type_d[0][1])],
                  self.structure_settings['structure_layer']['components_' + str(type_d[0][2])],
                  self.structure_settings['structure_layer']['components_' + str(type_d[0][3])]],\
                 [self.structure_settings['structure_layer']['components_' + str(type_d[1][0])],
                  self.structure_settings['structure_layer']['components_' + str(type_d[1][1])],
                  self.structure_settings['structure_layer']['components_' + str(type_d[1][2])],
                  self.structure_settings['structure_layer']['components_' + str(type_d[1][3])]]
        return type_s

    def get_encoding_structure(self):
        return self.structure_settings['neurons_encoding']

    def get_parameters_reservoir(self):
        parameter = self.get_sub_dict(self.decode('Reservoir'), 'plasticity', 'strength')
        return parameter

    def get_parameters_encoding_readout(self):
        parameter = self.decode('Encoding_Readout')
        return parameter

    def get_parameters_initialization(self):
        parameters = {}
        parameters_block_neurons = {}
        parameters_block_synapses = {}
        block_types = self.get_reservoir_block_type()
        for block_type in block_types:
            parameters_block_neurons[block_type] = self.get_sub_dict(self.block_decoder_type[block_type]('parameter'),
                                                                     'tau', 'threshold')
            parameters_block_synapses[block_type] = self.get_sub_dict(self.block_decoder_type[block_type]('parameter'),
                                                                      'plasticity', 'strength', 'type')

        parameters['reservoir'] = {'parameter_block_neurons':parameters_block_neurons,
                                   'parameter_block_synapses':parameters_block_synapses,
                                   'parameter_pathway': self.get_parameters_reservoir()}
        parameters['encoding'] = None
        parameters['readout'] = None
        parameters['encoding_reservoir'] = self.get_parameters_encoding_readout()
        parameters['reservoir_readout'] = None
        return parameters