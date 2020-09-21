# -*- coding: utf-8 -*-
"""
    The components generator based on the parameter
    decoded from optimizer.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""


from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.networks.components import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import *

from brian2 import *


class Generator_connection_matrix(BaseFunctions):
    def __init__(self, random_state):
        super().__init__()
        self.random_state = random_state
        self.block_generator_type = {'random':self.generate_connection_matrix_random,
                                   'scale_free':self.generate_connection_matrix_scale_free,
                                   'circle':self.generate_connection_matrix_circle,
                                   'hierarchy':self.generate_connection_matrix_hierarchy}

    def generate_connection_matrix_random(self, N, p):
        connection_matrix_out, connection_matrix_in = self.full_connected(N, p)
        return np.array([connection_matrix_out, connection_matrix_in])

    def generate_connection_matrix_scale_free(self, N, p_alpha, p_beta, p_gama):
        alpha = p_alpha/sum(p_alpha+p_beta+p_gama)
        beta = p_beta/sum(p_alpha+p_beta+p_gama)
        gama = p_gama/sum(p_alpha+p_beta+p_gama)
        DSF = Direct_scale_free(final_nodes = N,  alpha = alpha, beta = beta, gama = gama,
                                init_nodes = 1, detla_in=1, detla_out = 1)
        DSF.generate_gaph()
        connection_matrix_out, connection_matrix_in = DSF.o, DSF.i
        return np.array([connection_matrix_out, connection_matrix_in])

    def generate_connection_matrix_circle(self, N, p_forward, p_backward, p_threshold):
        connection_matrix_out, connection_matrix_in = [], []
        nodes = np.arange(N)
        start = np.random.randint(0, N)
        circle = []
        for i in range(N):
            try:
                circle.append(nodes[start + i])
            except IndexError:
                circle.append(nodes[start + i -N])
        circle = circle + circle

        for index_pre, node_pre in enumerate(circle[:N]):
            for index_post, node_post in enumerate(circle[index_pre+1:N+index_pre+1]):
                distance = index_post
                decay = (N-distance-1)/(N-1)
                if np.random.rand() <= p_forward * (decay - p_threshold):
                    connection_matrix_out.append(node_pre)
                    connection_matrix_in.append(node_post)
                if np.random.rand() <= p_backward * (decay - p_threshold):
                    connection_matrix_out.append(node_post)
                    connection_matrix_in.append(node_pre)
        return np.array(connection_matrix_out, connection_matrix_in)

    def generate_connection_matrix_hierarchy(self, N_i, N_h, N_o, p_out, p_in, decay):
        connection_matrix_out, connection_matrix_in = [], []
        nodes = np.arange(N_i + N_h + N_o)
        nodes_ = [nodes[:N_i], nodes[N_i:N_h + N_i], nodes[N_h + N_i:N_i + N_h + N_o]]
        p_out_ = [np.array([p_out] * N_i), np.array([p_out] * N_h), np.array([p_out] * N_o)]
        p_in_ = [np.array([p_in] * N_i), np.array([p_in] * N_h), np.array([p_in] * N_o)]
        circle = [0, 1, 2] + [0, 1, 2]
        for i in circle[:3]:
            nodes_mid, p_out_mid, p_in_mid = nodes_[circle[i]], p_out_[circle[i]], p_in_[circle[i]]
            nodes_pre, p_out_pre, p_in_pre = nodes_[circle[i - 1]], p_out_[circle[i - 1]], p_in_[circle[i - 1]]
            nodes_post, p_out_post, p_in_post = nodes_[circle[i + 1]], p_out_[circle[i + 1]], p_in_[circle[i + 1]]
            for out_mid_index, node in enumerate(nodes_mid):
                in_post, in_pre = p_in_post.argsort()[::-1], p_in_pre.argsort()[::-1]
                for in_post_index, in_pre_index in zip(in_post, in_pre):
                    if np.random.rand() <= p_out_mid[out_mid_index]  \
                            and np.random.rand() <= p_out_post[in_post_index]:
                        connection_matrix_out.append(node)
                        connection_matrix_in.append(nodes_post[in_post_index])
                        p_out_mid[out_mid_index] = p_out_mid[out_mid_index] * decay
                        p_in_post[in_post_index] = p_in_post[in_post_index] * decay
                    if np.random.rand() <= p_out_mid[out_mid_index] \
                            and np.random.rand() <= p_out_pre[in_pre_index]:
                        connection_matrix_out.append(node)
                        connection_matrix_in.append(nodes_pre[in_pre_index])
                        p_out_mid[out_mid_index] = p_out_mid[out_mid_index] * decay
                        p_out_pre[in_pre_index] = p_out_pre[in_pre_index] * decay
        return np.array(connection_matrix_out, connection_matrix_in)

    def generate_connection_matrix_reservoir_layer(self, blocks_type, count, layer, structure_type, cmo, cmi):
        cmo_, cmi_ = cmo, cmi
        count_ = count
        blocks_type_ = blocks_type
        if layer > 0 :
            o, i = [],[]
            for name in structure_type[layer]:
                blocks_type_, count_, cmo_, cmi_, o_, i_ = \
                    self.generate_connection_matrix_reservoir_layer(blocks_type_, count_, layer-1, structure_type, cmo_, cmi_)
                cmo_.extend(list(np.array(o_)[structure_layer[name]['structure'][0]].reshape(-1)))
                cmi_.extend(list(np.array(i_)[structure_layer[name]['structure'][1]].reshape(-1)))
                o.append(list(np.array(o_)[structure_layer[name]['output_input'][0]].reshape(-1)))
                i.append(list(np.array(i_)[structure_layer[name]['output_input'][1]].reshape(-1)))
            return blocks_type_, count_, cmo_, cmi_, o, i
        else:
            o, i = [],[]
            for name in structure_type[layer]:
                cmo_.extend(list(np.array(structure_layer[name]['structure'][0]) + count_))
                cmi_.extend(list(np.array(structure_layer[name]['structure'][1]) + count_))
                o.append(list(np.array(structure_layer[name]['output_input'][0]) + count_))
                i.append(list(np.array(structure_layer[name]['output_input'][1]) + count_))
                n = max(structure_layer[name]['structure'][0] + structure_layer[name]['structure'][1]) + 1
                count_ = count_ + n
                blocks_type_.extend(list(np.arange(n)))
            return blocks_type_, count_, cmo_, cmi_, o, i

    def generate_connection_matrix_reservoir(self, structure_type):
        # structure_type = [['components_1','components_2','components_1','components_2'],
        # ['components_3','components_4','components_3','components_4']]
        connection_matrix_out, connection_matrix_in = [], []
        layer = len(structure_type)-1
        count = 0
        blocks_type = []
        o, i = [], []
        blocks_type, count, connection_matrix_out, connection_matrix_in, o_, i_ = \
            self.generate_connection_matrix_reservoir_layer(blocks_type, count, layer, structure_type,
                                                  connection_matrix_out, connection_matrix_in)
        connection_matrix_out.extend(list(np.array(o_)[structure_reservoir['components']['structure'][0]].reshape(-1)))
        connection_matrix_in.extend(list(np.array(i_)[structure_reservoir['components']['structure'][1]].reshape(-1)))
        o.append(list(np.array(o_)[structure_reservoir['components']['output_input'][0]].reshape(-1)))
        i.append(list(np.array(i_)[structure_reservoir['components']['output_input'][1]].reshape(-1)))
        return blocks_type, np.array(connection_matrix_out, connection_matrix_in), o, i


class Generator(Generator_connection_matrix):
    '''
     Initialize the parameters of the neurons or synapses.

     Parameters
     ----------
     # some public used such as random_state.
     '''

    def __init__(self, random_state):
        super().__init__(random_state)

    def register_decoder(self, decoder):
        self.decoder = decoder

    def generate_block(self, name, get_parameter_structure, get_matrix):
        parameter_structure = get_parameter_structure()
        connect_matrix = get_matrix('structure', parameter_structure)
        block = Block(parameter_structure['N'], connect_matrix)
        block.create_neurons(dynamics_reservoir, threshold = threshold_reservoir, reset = reset_reservoir,
                             refractory = refractory_reservoir, name='block_' + name)
        block.create_synapse(dynamics_synapse, dynamics_synapse_pre,
                             name='block_block_' + name)
        block.separate_ex_inh()
        block.connect()
        block.determine_input_output()
        return block

    def generate_blocks(self, block_type, blocks_type):
        block_group = BlockGroup()
        for index, block in enumerate(blocks_type):
            type = block_type[block]
            block_decoder = self.block_decoder_type[type]
            block_generator = self.block_generator_type[type]
            block = self.generate_block(type + '_' + str(index), block_decoder, block_generator)
            block_group.add_block(block, type)
        return block_group

    def generate_pathway(self, name, pre_group, post_group, connection_matrix, model, model_pre, model_post):
        pathway = Pathway(pre_group.blocks, post_group.blocks, connection_matrix)
        pathway.create_synapse(model, model_pre, model_post,  name = name)
        return pathway

    def generate_reservoir(self):
        reservoir = Reservoir()
        block_type = self.decoder.get_reservoir_block_type()
        structure_type = self.decoder.get_reservoir_structure_type()
        blocks_type, connection_matrix, o, i = self.generate_connection_matrix_reservoir(structure_type)
        block_group = self.generate_blocks(block_type, blocks_type)
        pathway = self.generate_pathway('pathway_reservoir_', block_group, block_group, connection_matrix,
                                        dynamics_synapse_STDP, dynamics_synapse_pre_STDP,
                                        dynamics_synapse_post_STDP)
        reservoir.register_blocks(block_group)
        reservoir.register_pathway(pathway)
        reservoir.register_input_output(o, i)
        reservoir.connect()
        return reservoir

    def generate_encoding(self):
        block_group = BlockGroup()
        N = self.decoder.get_encoding_structure()
        block = Block(N, np.array([],[]))
        block.create_neurons(dynamics_encoding, threshold='I > 0', reset = '0',
                             refractory = 0 * ms , name='block_encoding')
        block.determine_input_output()
        block_group.add_block(block)
        return block_group

    def generate_readout(self):
        block_group = BlockGroup()
        N = self.decoder.get_encoding_structure()
        block = Block(N, np.array([],[]))
        block.create_neurons(dynamics_readout, threshold=None, reset = None,
                             refractory = None, name='block_readout')
        block.determine_input_output()
        block_group.add_block(block)
        return block_group

    def generate_pathway_encoding_reservoir(self, encoding, reservoir):
        connection_matrix = [[0]*len(reservoir.input), reservoir.input]
        pathway = self.generate_pathway('pathway_encoding_', encoding, reservoir.block_group, connection_matrix,
                                        dynamics_synapse, dynamics_synapse_pre, None)
        return pathway

    def generate_pathway_reservoir_readout(self, reservoir, readout):
        connection_matrix = [reservoir.output, [0]*len(reservoir.output)]
        pathway = self.generate_pathway('pathway_readout_', reservoir.block_group, readout, connection_matrix,
                                        'w = 1 : 1', dynamics_synapse_pre, None)
        return pathway

    def generate_network(self):
        network = LSM_Network()
        encoding = self.generate_encoding()
        reservoir = self.generate_reservoir()
        readout = self.generate_readout()
        pathway_encoding_reservoir = self.generate_pathway_encoding_reservoir(encoding, reservoir)
        pathway_reservoir_readout = self.generate_pathway_reservoir_readout(reservoir, readout)

        network.register_layer(encoding, 'encoding')
        network.register_layer(reservoir, 'reservoir')
        network.register_layer(readout, 'readout')
        network.register_pathway(pathway_encoding_reservoir, 'encoding_reservoir')
        network.register_pathway(pathway_reservoir_readout, 'reservoir_readout')
        network.connect()
        return network

    def initialize(self, network):
        parameters = self.decoder.get_parameters_initialization()
        network.initialize(parameters)

    def generate_and_initialize(self):
        network = self.generate_network()
        self.initialize(network)
        return network