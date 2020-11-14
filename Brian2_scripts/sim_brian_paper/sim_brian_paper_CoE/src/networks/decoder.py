# -*- coding: utf-8 -*-
"""
    The decoding method between genotype
    and the parameters.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.config import structure_blocks, voltage_reset
import numpy as np

class Decoder(BaseFunctions):
    """
    This class offers the decoding function which convert the gen into the
    parameters used by generator.

    Parameters
    ----------
    config_group, config_keys, config_SubCom, config_codes, config_ranges, config_borders,
    config_precisions, config_scales: list, the config of optimization methods.
    neurons_encoding: the number of encoding neurons .
    """

    def __init__(self, config_group, config_keys, config_SubCom, config_codes, config_ranges, config_borders,
                      config_precisions, config_scales, neurons_encoding):
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
        self.neurons_encoding = neurons_encoding

    @property
    def get_keys(self):
        '''
         Get the name of each variate.

         Parameters
         ----------
         '''

        keys = []
        for index, group_name in enumerate(self.config_group):
            for key in self.config_keys[index]:
                keys.append(group_name+'_'+key)
        return keys

    @property
    def get_SubCom(self):
        '''
         Get the SubCom in the config.

         Parameters
         ----------
         '''

        return self.config_SubCom

    @property
    def get_dim(self):
        '''
         Get the total dimension of the  variate.

         Parameters
         ----------
         '''

        return len(self.get_keys)

    @property
    def get_codes(self):
        '''
         Get the settings of encoding of the variate.

         Parameters
         ----------
         '''

        codes = []
        for group_codes in self.config_codes:
            codes.extend(group_codes)
        return np.array(codes)

    @property
    def get_ranges(self):
        '''
         Get the settings of ranges of the variate.

         Parameters
         ----------
         '''

        ranges = []
        for group_ranges in self.config_ranges:
            ranges.extend(group_ranges)
        return np.array(ranges).T

    @property
    def get_borders(self):
        '''
         Get the settings of borders of the variate.

         Parameters
         ----------
         '''

        borders = []
        for group_borders in self.config_borders:
            borders.extend(group_borders)
        return np.array(borders).T

    @property
    def get_precisions(self):
        '''
         Get the settings of precisions of the variate.

         Parameters
         ----------
         '''

        precisions = []
        for group_precisions in self.config_precisions:
            precisions.extend(group_precisions)
        return np.array(precisions)

    @property
    def get_scales(self):
        '''
         Get the settings of scales of the variate.

         Parameters
         ----------
         '''
        scales = []
        for group_scales in self.config_scales:
            scales.extend(group_scales)
        return np.array(scales)

    def _float2int(self, Gen):
        '''
         Register a gen for this decoder, which is ready to be decode for the generator.

         Parameters
         ----------
         Gen: np.array, the gen generated by optimization algorithm with all float
         '''

        _Gen = []
        for index, p in enumerate(self.get_precisions):
            if p == 0:
                _Gen.append(int(Gen[index]))
            else:
                _Gen.append(Gen[index])
        return _Gen

    def register(self, Gen):
        '''
         Register a gen for this decoder, which is ready to be decode for the generator.

         Parameters
         ----------
         Gen: list, the gen generated by optimization algorithm but the precision with '0' convert to int
         '''

        self.Gen = self._float2int(Gen)

    def decode(self, target):
        '''
         Decode the target variate from the registered gen.

         Parameters
         ----------
         target: str, the name of target variate.
         '''

        group = np.where(np.array(self.config_group) == target)[0][0]
        key = self.config_keys[group]
        SubCom = self.config_SubCom[group]
        codes = self.config_codes[group]
        ranges = self.config_ranges[group]
        parameter = {}
        for p,k,c,r in zip(self.sub_list(self.Gen, SubCom), key, codes, ranges):
            if c != None:
                l = len(self.dec2bin((r[1]-r[0]), 0))
                p = self.dec2bin(p, l)
            parameter[k] = p
        return parameter

    def decode_block_random(self, need):
        '''
         Decode the information of the random block.

         Parameters
         ----------
         need: str, 'structure' or 'parameter'.
         '''

        parameters = self.decode('Block_random')
        if need == 'structure':
            sub_parameters = self.get_sub_dict(parameters, 'N', 'p')
            return sub_parameters
        if need == 'parameter':
            sub_parameters = self.get_sub_dict(parameters, 'plasticity', 'strength', 'tau', 'threshold', 'type')
            return sub_parameters

    def decode_block_scale_free(self, need):
        '''
         Decode the information of the scale free block.

         Parameters
         ----------
         need: str, 'structure' or 'parameter'.
         '''

        parameters = self.decode('Block_scale_free')
        if need == 'structure':
            sub_parameters = self.get_sub_dict(parameters, 'N', 'p_alpha', 'p_beta', 'p_gama')
            return sub_parameters
        if need == 'parameter':
            sub_parameters = self.get_sub_dict(parameters, 'plasticity', 'strength', 'tau', 'threshold', 'type')
            return sub_parameters

    def decode_block_circle(self, need):
        '''
         Decode the information of the circle block.

         Parameters
         ----------
         need: str, 'structure' or 'parameter'.
         '''

        parameters = self.decode('Block_circle')
        if need == 'structure':
            sub_parameters = self.get_sub_dict(parameters, 'N', 'p_backward', 'p_forward', 'p_threshold')
            return sub_parameters
        if need == 'parameter':
            sub_parameters = self.get_sub_dict(parameters, 'plasticity', 'strength', 'tau', 'threshold', 'type')
            return sub_parameters

    def decode_block_hierarchy(self, need):
        '''
         Decode the information of the hierarchy block.

         Parameters
         ----------
         need: str, 'structure' or 'parameter'.
         '''

        parameters = self.decode('Block_hierarchy')
        if need == 'structure':
            sub_parameters = self.get_sub_dict(parameters, 'N_h', 'N_i', 'N_o', 'decay', 'p_in', 'p_out')
            return sub_parameters
        if need == 'parameter':
            sub_parameters = self.get_sub_dict(parameters, 'plasticity', 'strength', 'tau', 'threshold', 'type')
            return sub_parameters

    def get_reservoir_block_type(self):
        '''
         Decode the block type of each position of basic block group.

         Parameters
         ----------
         '''

        parameter = self.decode('Reservoir_config')
        type_b = parameter['block']
        type_d, i = [], 0
        while True:
            temp = type_b[i:i + 2]
            i += 2
            if len(temp) == 2:
                type_d.append(self.bin2dec(temp))
            else:
                return type_d

    def get_reservoir_structure_type(self):
        '''
         Decode the block group type of each position of each layer.

         Parameters
         ----------
         '''

        parameter = self.decode('Reservoir_config')
        type_d, layer = [], 1
        while True:
            try:
                type_b = parameter['layer_' + str(layer)]
                type_d_i, i = [], 0
                while True:
                    temp = type_b[i:i + 2]
                    i += 2
                    if len(temp) == 2:
                        type_d_i.append(self.bin2dec(temp))
                    else:
                        type_d.append(type_d_i)
                        break
                layer += 1
            except KeyError:
                return type_d

    def get_parameters_reservoir(self):
        '''
         Decode the parameters of the reservoir.

         Parameters
         ----------
         '''

        parameter = self.get_sub_dict(self.decode('Reservoir_config'), 'plasticity', 'strength', 'type')
        return parameter

    def get_parameters_encoding_readout(self):
        '''
         Decode parameters of the encoding_readout.

         Parameters
         ----------
         '''

        parameter = self.decode('Encoding_Readout')
        return parameter

    def get_encoding_structure(self):
        '''
         Decode the encoding structure, the parameters are come from the need of dataloader.

         Parameters
         ----------
         '''

        return self.neurons_encoding

    def get_parameters_initialization(self):
        '''
         Decoder and organize the parameters of initialization for the
         whole LSM_network.

         Parameters
         ----------
         '''

        parameters = {}
        parameters_block_neurons = {}
        parameters_block_synapses = {}
        block_types = self.get_reservoir_block_type()
        for block_type in block_types:
            name = structure_blocks['components_'+str(block_type)]
            parameters_block_neurons[name] = self.get_sub_dict(self.block_decoder_type[name]('parameter'),
                                                                     'tau', 'threshold')
            parameters_block_neurons[name]['v'] = voltage_reset
            parameters_block_synapses[name] = self.get_sub_dict(self.block_decoder_type[name]('parameter'),
                                                                      'plasticity', 'strength', 'type')
            self.change_dict_key(parameters_block_synapses[name],'strength','strength_need_random')

        parameters['reservoir'] = {'parameter_block_neurons':parameters_block_neurons,
                                   'parameter_block_synapses':parameters_block_synapses,
                                   'parameter_pathway': self.get_parameters_reservoir()}
        self.change_dict_key(parameters['reservoir']['parameter_pathway'], 'strength', 'strength_need_random')
        parameters['encoding'] = None
        parameters['readout'] = None
        parameters['encoding_reservoir'] = self.get_parameters_encoding_readout()
        self.change_dict_key(parameters['encoding_reservoir'], 'strength', 'strength_need_random')
        parameters['reservoir_readout'] = None
        return parameters