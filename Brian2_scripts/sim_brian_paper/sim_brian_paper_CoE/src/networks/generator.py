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


class Generator(BaseFunctions):
    '''
     Initialize the parameters of the neurons or synapses.

     Parameters
     ----------
     # some public used such as random_state.
     '''

    def __init__(self, random_state):
        super().__init__()
        self.random_state = random_state

    def register_decoder(self, decoder):
        self.decoder = decoder

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

    def generate_connection_matrix_circle(self, N, p_forward, p_backward, threshold):
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
                if np.random.rand() <= p_forward * (decay - threshold):
                    connection_matrix_out.append(node_pre)
                    connection_matrix_in.append(node_post)
                if np.random.rand() <= p_backward * (decay - threshold):
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

    def generate_connection_matrix_reservoir(self):

        connection_matrix_out, connection_matrix_in = [], []
        return np.array(connection_matrix_out, connection_matrix_in)


    def generate_block(self, name, get_parameter_structure, get_matrix):
        parameter_structure = get_parameter_structure()
        connect_matrix = get_matrix(parameter_structure)
        block = Block(parameter_structure['N'], connect_matrix)
        block.create_neurons(dynamics_reservoir, threshold = threshold_reservoir, reset = reset_reservoir,
                             refractory = refractory_reservoir, name='block_' + name)
        block.create_synapse(dynamics_synapse, dynamics_synapse_pre,
                             name='block_block_' + name)
        block.separate_ex_inh()
        block.connect()
        block.determine_input_output()
        return block

    def generate_blocks(self, N1, N2, N3, N4):
        block_group = BlockGroup()
        for index in range(N1):
            block = self.generate_block('random_' + str(index), self.decoder.decode_block_random,
                            self.generate_connection_matrix_random)
            block_group.add_block(block)
        for index in range(N2):
            block = self.generate_block('scale_free_' + str(index + N1), self.decoder.decode_block_scale_free,
                            self.generate_connection_matrix_scale_free)
            block_group.add_block(block)
        for index in range(N3):
            block = self.generate_block('circle_' + str(index + N1 + N2), self.decoder.decode_block_circle,
                            self.generate_connection_matrix_circle)
            block_group.add_block(block)
        for index in range(N4):
            block = self.generate_block('hierarchy_' + str(index + N1 + N2 + N3), self.decoder.decode_block_hierarchy,
                                        self.generate_connection_matrix_hierarchy)
            block_group.add_block(block)
        return block_group

    def generate_pathway_reservoir(self, Block_group):
        connection_matrix = self.generate_connection_matrix_reservoir()
        pathway = Pathway(Block_group.blocks, BaseFunctions.blocks, connection_matrix)
        pathway.create_synapse(dynamics_synapse_STDP, dynamics_synapse_pre_STDP,
                               dynamics_synapse_post_STDP,  name = 'pathway_reservoir_')
        return pathway

    def generate_reservoir(self):
        reservoir = Reservoir()
        N1, N2, N3, N4 = self.decoder.get_reservoir()
        block_group = self.generate_blocks(N1, N2, N3, N4)
        pathway = self.generate_pathway_reservoir(block_group)
        reservoir.register_blocks(block_group)
        reservoir.register_pathway(pathway)
        reservoir.determine_input_output()
        reservoir.connect()
        return reservoir

    def generate_encoding(self):
        block_group = BlockGroup()
        return block_group

    def generate_readout(self):
        block_group = BlockGroup()
        return block_group

    def generate_pathway_encoding_reservoir(self, encoding, reservoir):
        connection_matrix = [[0], reservoir.input]
        pathway = Pathway(encoding.blocks, reservoir.block_group.blocks, connection_matrix)
        pathway.create_synapse(dynamics_synapse, dynamics_synapse_pre,
                               None,  name = 'pathway_encoding_')
        return pathway

    def generate_pathway_reservoir_readout(self, reservoir, readout):
        connection_matrix = [reservoir.output, [0]]
        pathway = Pathway(reservoir.block_group.blocks, readout.blocks, connection_matrix)
        pathway.create_synapse('w = 1 : 1', dynamics_synapse_pre,
                               None,  name = 'pathway_readout_')
        return pathway

    def generate_network(self):
        network = LSM_Network()
        encoding = self.generate_encoding()
        reservoir = self.generate_reservoir()
        readout = self.generate_readout()
        pathway_encoding_reservoir = self.generate_pathway_encoding_reservoir()
        pathway_reservoir_readout = self.generate_pathway_reservoir_readout()

        network.register_layer(encoding, 'encoding')
        network.register_layer(reservoir, 'reservoir')
        network.register_layer(readout, 'readout')
        network.register_pathway(pathway_encoding_reservoir, 'encoding_reservoir')
        network.register_pathway(pathway_reservoir_readout, 'reservoir_readout')
        network.connect()

        parameters = self.decoder.get_parameters_network()
        network.initialize(parameters)

        return network



    # def create_blocks(self, neurons_block, connect_matrix_blocks, dynamics_neurons, dynamics_synapse,
    #                   dynamics_synapse_pre, threshold, reset, refractory):
    #     '''
    #      Create blocks for the reservoir.
    #
    #      Parameters
    #      ----------
    #      neurons_block: int, the number of neurons in each block.
    #      connect_matrix_blocks: list, a list of connect_matrix for the block.
    #      Other parameters follow the necessary 'Synapses' class of Brain2.
    #      '''
    #
    #     for index, connect_matrix in zip(range(self.N), connect_matrix_blocks):
    #         block_reservoir = Block(neurons_block)
    #         block_reservoir.create_neurons(dynamics_neurons, threshold = threshold, reset = reset,
    #                                        refractory = refractory, name='block_' + str(index))
    #         block_reservoir.create_synapse(dynamics_synapse, dynamics_synapse_pre,
    #                                        name='block_block_' + str(index))
    #         block_reservoir.connect(connect_matrix)
    #         self.add_block(block_reservoir)
    #
    # def create_synapse(self, connect_matrix, model, on_pre, **kwargs):
    #     '''
    #      Create synapses between blocks.
    #
    #      Parameters
    #      ----------
    #      connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse between blocks.
    #                     The first list is the pre-synapse blocks and the second list is the post-synapse blocks.
    #      Other parameters follow the necessary 'Synapses' class of Brain2.
    #      '''
    #
    #     self.pathway = Pathway(self.blocks, connect_matrix, model, on_pre, name = 'reservoir_', **kwargs)
    #
    #
    # def initialize_blocks_neurons(self, **kwargs):
    #     '''
    #      Initialize the parameters of the neurons of blocks in the reservoir.
    #
    #      Parameters
    #      ----------
    #      kwargs: dict{key:str(name), value:np.array[n*1]}, extensible dict of parameters.
    #      '''
    #
    #     for block_reservoir in self.blocks:
    #         block_reservoir.initialize_neurons(**kwargs)
    #
    # def initialize_blocks_synapses(self, **kwargs):
    #     '''
    #      Initialize the parameters of the synapses of blocks in the reservoir.
    #
    #      Parameters
    #      ----------
    #      kwargs: dict{key:str(name), value:np.array[n*n]}, extensible dict of parameters.
    #      '''
    #
    #     for block_reservoir in self.blocks:
    #         block_reservoir.initialize_synapses(**kwargs)
    #
    # def initialize_reservoir_synapses(self, **kwargs):
    #     '''
    #      Initialize the parameters of the synapses between blocks in the reservoir.
    #
    #      Parameters
    #      ----------
    #      kwargs: dict{key:str(name), value:np.array[n*n]}, extensible dict of parameters.
    #      '''
    #
    #     for index, synapse in enumerate(self.synapses):
    #         for key, value in zip(kwargs.keys(), kwargs.values()):
    #             connect_matrix = [list(synapse.i), list(synapse.j)]
    #
    #             parameter_synapse = np.zeros(synapse.N_pre[0], synapse.N_post[0])
    #             parameter = value[self.output[index]][self.input[index]]
    #             parameter_ = np.random.rand(synapse.N[0] * synapse.N[0]) * parameter
    #             for index, (index_i, index_j) in enumerate(zip(connect_matrix)):
    #                 parameter_synapse[index_i][index_j] = parameter_[index]
    #
    #             converted_value = self.get_parameters(connect_matrix, parameter_synapse)
    #             self.initialize_parameters(synapse, key, converted_value)
    #
    #
    # def create_synapse_encoding(self, dynamics_synapse, dynamics_synapse_pre, reservoir_input):
    #     '''
    #      Create synapses between encoding and blocks.
    #
    #      Parameters
    #      ----------
    #      reservoir_input: int, the number of blocks in the reservoir used for input.
    #      Other parameters follow the necessary 'Synapses' class of Brain2.
    #      '''
    #
    #     self.layers['reservoir'].determine_input(reservoir_input)
    #     for index, block_reservoir_index in enumerate(self.layers['reservoir'].input):
    #         synapse_encoding_reservoir = Synapses(self.layers['encoding'],
    #                                               self.layers['reservoir'].blocks[block_reservoir_index].neurons,
    #                                               dynamics_synapse,
    #                                               on_pre=dynamics_synapse_pre,
    #                                               method='euler', name='encoding_block_' + str(index))
    #         self.synapses_encoding.append(synapse_encoding_reservoir)
    #
    # def create_synapse_readout(self, dynamics_synapse_pre):
    #     '''
    #      Create synapses between readout and blocks.
    #
    #      Parameters
    #      ----------
    #      Other parameters follow the necessary 'Synapses' class of Brain2.
    #      '''
    #
    #     for index, block_reservoir in enumerate(self.layers['reservoir'].blocks):
    #         synapse_reservoir_readout = Synapses(block_reservoir.neurons, self.layers['readout'], 'w = 1 : 1',
    #                                              on_pre = dynamics_synapse_pre, name='readout_' + str(index))
    #         self.synapses_readout.append(synapse_reservoir_readout)
    #
    #
    # def connect_encoding(self):
    #     '''
    #      Connect blocks and encoding base on the synapses in the network.
    #      The connection is full connection from the encoding layer to the input blocks.
    #
    #      Parameters
    #      ----------
    #      '''
    #
    #     for index, synapse_encoding_reservoir in enumerate(self.synapses_encoding):
    #         block_post_index = self.layers['reservoir'].input[index]
    #         block_post = self.layers['reservoir'].blocks[block_post_index]
    #         input = block_post.input
    #         output = np.arange(self.layers['encoding'].N[0])
    #         connect_matrix = self.np_two_combination(output, input)
    #         synapse_encoding_reservoir.connect(i = connect_matrix[0], j = connect_matrix[1]) # var 'input' needs to be test
    #
    # def connect_readout(self):
    #     '''
    #      Connect blocks and readout base on the synapses in the network.
    #      The connections is one to one.
    #
    #      Parameters
    #      ----------
    #      '''
    #
    #     neurons_block = self.layers['reservoir'].blocks[0].N
    #     for index, synapse_reservoir_readout in enumerate(self.synapses_readout):
    #         synapse_reservoir_readout.connect(j='i+' + str(index * neurons_block))
    #
    # def initialize_parameters_encoding_synapses(self, **kwargs):
    #     '''
    #      Initialize the parameters of the synapses between reservoir and encoding.
    #
    #      Parameters
    #      ----------
    #      kwargs: dict{key:str(name), value:np.array[n*1]}, extensible dict of parameters.
    #      '''
    #
    #     for index, synapse in enumerate(self.synapses_encoding):
    #         for key, value in zip(kwargs.keys(), kwargs.values()):
    #             connect_matrix = [list(synapse.i), list(synapse.j)]
    #
    #             parameter_synapse = np.zeros(synapse.N_pre[0], synapse.N_post[0])
    #             parameter = value[self.layers['reservoir'].input[index]]
    #             parameter_ = np.random.rand(synapse.N[0] * synapse.N[0]) * parameter
    #             for index, (index_i, index_j) in enumerate(zip(connect_matrix)):
    #                 parameter_synapse[index_i][index_j] = parameter_[index]
    #
    #             converted_value = self.get_parameters(connect_matrix, parameter_synapse)
    #             self.initialize_parameters(synapse, key, converted_value)
    #
    #
    # def create_encoding(self, dynamics_encoding):
    #     self.neurons = NeuronGroup(self.N, dynamics_encoding, threshold='I > 0', method='euler',
    #                            refractory=0 * ms, name='encoding')
    #
    # def create_readout(self, dynamics_readout):
    #     self.neurons = NeuronGroup(self.N, dynamics_readout, method='euler', name='readout')