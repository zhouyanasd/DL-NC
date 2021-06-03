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
    """
    This class offers the connection matrix generation functions
    of four kinds of blocks and reservoir.

    Parameters
    ----------
    random_state: int, random state generated by np.random.
    """

    def __init__(self, random_state):
        super().__init__()
        self.random_state = random_state
        self.block_types_N = len(list(structure_blocks))
        self.layer_types_N = len(list(structure_layer))

    def generate_connection_matrix_blocks(self, block_type, **parameters):
        '''
         General function of the generate connection matrix function.

         Parameters
         ----------
         block_type: int, the order of the block.
         parameters: dict, the connection parameters for different generation type.
         '''
        parameters_ = self.get_sub_dict(parameters, 'N')
        component = structure_blocks['components_' + str(block_type)]
        if component['name'] == 'random':
            parameters_['p'] = self.adapt_scale(component['p_0'], parameters['p_0'])
        elif component['name'] == 'scale_free':
            parameters_['p_alpha'] = self.adapt_scale(component['p_0'], parameters['p_0'])
            parameters_['p_beta'] = self.adapt_scale(component['p_1'], parameters['p_1'])
            parameters_['p_gama'] = self.adapt_scale(component['p_2'], parameters['p_2'])
        elif component['name'] == 'circle':
            parameters_['p_forward'] = self.adapt_scale(component['p_0'], parameters['p_0'])
            parameters_['p_backward'] = self.adapt_scale(component['p_1'], parameters['p_1'])
            parameters_['p_threshold'] = self.adapt_scale(component['p_2'], parameters['p_2'])
        elif component['name'] == 'hierarchy':
            parameters_['p_out'] = self.adapt_scale(component['p_0'], parameters['p_0'])
            parameters_['p_in'] = self.adapt_scale(component['p_1'], parameters['p_1'])
            parameters_['decay'] = self.adapt_scale(component['p_2'], parameters['p_2'])

        block_generator_type = {'random': self.generate_connection_matrix_blocks_random,
                                'scale_free': self.generate_connection_matrix_blocks_scale_free,
                                'circle': self.generate_connection_matrix_blocks_circle,
                                'hierarchy': self.generate_connection_matrix_blocks_hierarchy}
        return block_generator_type[ component['name']](**parameters_)

    def generate_connection_matrix_blocks_random(self, N, p):
        '''
         Generate connection matrix of random block

         Parameters
         ----------
         N: int, number of neurons of the block.
         p: double, the connection probability between two neurons.
         '''

        connection_matrix_out, connection_matrix_in = [], []
        for node_pre in np.arange(N):
            for node_post in np.arange(N):
                if node_pre == node_post:
                    continue
                elif np.random.rand() <= p:
                    connection_matrix_out.append(node_pre)
                    connection_matrix_in.append(node_post)
                else:
                    continue
        return N, np.array([connection_matrix_out, connection_matrix_in])

    def generate_connection_matrix_blocks_scale_free(self, N, p_alpha, p_beta, p_gama):
        '''
         Generate connection matrix of scale_free block

         Parameters
         ----------
         N: int, number of neurons of the block.
         p_alpha, p_beta, p_gama: double, the connection probability between two neurons.
         '''

        alpha = p_alpha/(p_alpha+p_beta+p_gama)
        beta = p_beta/(p_alpha+p_beta+p_gama)
        gama = p_gama/(p_alpha+p_beta+p_gama)
        DSF = Direct_scale_free(final_nodes = N,  alpha = alpha, beta = beta, gama = gama,
                                init_nodes = 1, delta_in=1, delta_out = 1)
        DSF.generate_graph()
        connection_matrix_out, connection_matrix_in = DSF.o, DSF.i
        return N, np.array([connection_matrix_out, connection_matrix_in])

    def generate_connection_matrix_blocks_circle(self, N, p_forward, p_backward, p_threshold):
        '''
         Generate connection matrix of circle block.
         'circle' is the critical variate to mark the position of neurons.
         'circle' is extend as 'circle + circle' to connect the tail and head.

         Parameters
         ----------
         N: int, number of neurons of the block.
         p_forward, p_backward, p_threshold: double, the connection probability between two neurons.
         '''

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
                decay = np.clip((N * p_threshold - distance - 1), 0.0, N) / (N * p_threshold - 1)
                if np.random.rand() <= p_forward * decay:
                    connection_matrix_out.append(node_pre)
                    connection_matrix_in.append(node_post)
                if np.random.rand() <= p_backward * decay:
                    connection_matrix_out.append(node_post)
                    connection_matrix_in.append(node_pre)
        return N, np.array([connection_matrix_out, connection_matrix_in])

    def generate_connection_matrix_blocks_hierarchy(self, N, p_out, p_in, decay):
        '''
         Generate connection matrix of hierarchy block.
         The hierarchy structure separate as three layer.

         Parameters
         ----------
         N_i, N_h, N_o: int, number of neurons of the block for different layer.
         p_out, p_in: double, the connection probability between two neurons.
         decay: double, the decay of connection probability.
         '''

        N_i, N_o = int(np.floor(N/3)), int(np.floor(N/3))
        N_h = N - N_i - N_o

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
                for in_post_index in in_post:
                    if np.random.rand() <= p_out_mid[out_mid_index] \
                            and np.random.rand() <= p_out_post[in_post_index]:
                        connection_matrix_out.append(node)
                        connection_matrix_in.append(nodes_post[in_post_index])
                        p_out_mid[out_mid_index] = p_out_mid[out_mid_index] * decay
                        p_in_post[in_post_index] = p_in_post[in_post_index] * decay
                for in_pre_index in in_pre:
                    if np.random.rand() <= p_out_mid[out_mid_index] \
                            and np.random.rand() <= p_out_pre[in_pre_index]:
                        connection_matrix_out.append(node)
                        connection_matrix_in.append(nodes_pre[in_pre_index])
                        p_out_mid[out_mid_index] = p_out_mid[out_mid_index] * decay
                        p_out_pre[in_pre_index] = p_out_pre[in_pre_index] * decay
        return N, np.array([connection_matrix_out, connection_matrix_in])

    def generate_connection_matrix_reservoir_layer(self, blocks_position, count, layer, structure_type, cmo, cmi):
        '''
         Generate connection matrix of reservoir for signal layer.
         This programme use recursion.

         Parameters
         ----------
         blocks_position: list, the existing block position in a basic block group.
                          A basic block group contain four block with different type decided by gen.
         count: int, the current number of neurons have been added in this reservoir.
                     it will help with the index of neurons.
         layer: int, the layer order.
         structure_type: tuple(list, list,...), the represent number of the structure of each layer.
         cmo: list, the 'connection_matrix[0]'.
         cmi: list, the 'connection_matrix[1]'.
         '''

        cmo_, cmi_ = cmo, cmi
        count_ = count
        blocks_position_ = blocks_position
        if layer > 0 :
            o, i = [],[]
            for gen in structure_type[layer]:
                component = structure_layer['components_' + str(gen)]
                blocks_position_, count_, cmo_, cmi_, o_, i_ = \
                    self.generate_connection_matrix_reservoir_layer(
                        blocks_position_, count_, layer-1, structure_type, cmo_, cmi_)
                for com_so, com_si in zip(component['structure'][0], component['structure'][1]):
                    for com_so_ in o_[com_so]:
                        for com_si_ in i_[com_si]:
                            cmo_.append(com_so_)
                            cmi_.append(com_si_)
                for com_o, com_i in zip(component['output_input'][0], component['output_input'][1]):
                    o.append(o_[com_o])
                    i.append(i_[com_i])
            return blocks_position_, count_, cmo_, cmi_, o, i
        else:
            o, i = [],[]
            for gen in structure_type[layer]:
                component = structure_layer['components_' + str(gen)]
                cmo_.extend(list(np.array(component['structure'][0]) + count_))
                cmi_.extend(list(np.array(component['structure'][1]) + count_))
                o.append(list(np.array(component['output_input'][0]) + count_))
                i.append(list(np.array(component['output_input'][1]) + count_))
                position = list(np.unique(component['structure'][0] + component['structure'][1] +
                                      component['output_input'][0] + component['output_input'][1]))
                count_ = count_ + len(position)
                blocks_position_.extend(position)
            return blocks_position_, count_, cmo_, cmi_, o, i

    def generate_connection_matrix_reservoir(self, structure_type):
        '''
         Generate connection matrix of reservoir, especially for the first four components of reservoir.
         This programme use recursion.

         Parameters
         ----------
         structure_type: tuple(list, list,...), the represent number of the structure of each layer.

         Example
         ----------
         structure_type = ([2, 0, 0, 1], [3, 3, 2, 3])
         '''

        connection_matrix_out, connection_matrix_in = [], []
        layer = len(structure_type)-1
        count = 0
        blocks_position = []
        o, i = [], []
        component = structure_reservoir['components']
        blocks_position, count, connection_matrix_out, connection_matrix_in, o_, i_ = \
            self.generate_connection_matrix_reservoir_layer(blocks_position, count, layer, structure_type,
                                                  connection_matrix_out, connection_matrix_in)
        for com_so, com_si in zip(component['structure'][0], component['structure'][1]):
            for com_so_ in o_[com_so]:
                for com_si_ in i_[com_si]:
                    connection_matrix_out.append(com_so_)
                    connection_matrix_in.append(com_si_)
        for com_o, com_i in zip(component['output_input'][0], component['output_input'][1]):
            o.extend(o_[com_o])
            i.extend(i_[com_i])
        return blocks_position, np.array([connection_matrix_out, connection_matrix_in]), o, i


class Generator(Generator_connection_matrix):
    """
    This class offers a basic generation functions of each components in network.

    Parameters
    ----------
    random_state: int, random state generated by np.random.
    """

    def __init__(self, random_state):
        super().__init__(random_state)

    def register_decoder(self, decoder):
        '''
         Add gen decoder to this generator.

         Parameters
         ----------
         decoder: Decoder, a instance of Decoder class.
         '''

        self.decoder = decoder

    def generate_block(self, index, position):
        '''
         A basic block generator function.

         Parameters
         ----------
         index: int, the block index of all blocks in reservoir.
         position: int, the block order of the block group.s
         '''

        block_type =  self.decoder.get_block_type(position)
        parameter_structure = self.decoder.get_block_structure(position)
        component_name = structure_blocks['components_' + str(block_type)]['name'] + '_' + str(index)
        N, connect_matrix = self.generate_connection_matrix_blocks(block_type, **parameter_structure)
        block = Block(N, connect_matrix)
        block.create_neurons(dynamics_reservoir, threshold = threshold_reservoir, reset = reset_reservoir,
                             refractory = refractory_reservoir, name='block_' + component_name)
        block.create_synapse(dynamics_block_synapse_STDP, dynamics_block_synapse_pre_STDP,
                            dynamics_block_synapse_post_STDP, name='block_block_' + component_name)
        block.separate_ex_inh()
        block.connect()
        block.determine_input_output()
        return block

    def generate_pathway(self, name, pre_group, post_group, connection_matrix, model, model_pre, model_post):
        '''
         A basic pathway generator function between the block group.

         Parameters
         ----------
         name: str, the pre name of the generated synapses.
         pre_group: BlockGroup, the block group before the pathway.
         post_group: post_group, the block group after the pathway.
         connection_matrix: list[list[int], list[int]], the fixed connection matrix between 'Block_group',
                            int for 'Block'.
         '''

        pathway = Pathway(pre_group.blocks, post_group.blocks, connection_matrix)
        pathway.create_synapse(model, model_pre, model_post,  name = name)
        return pathway

    def generate_blocks(self, blocks_position):
        '''
         Generate blocks for reservoir according to blocks_type,
         the blocks belong to one block group.

         Parameters
         ----------
         blocks_position: list[int], the blocks posidtion order according to 'structure_blocks'.
         '''

        block_group = BlockGroup()
        for index, position in enumerate(blocks_position):
            block = self.generate_block(index, position)
            block_group.add_block(block, position)
        return block_group

    def generate_reservoir(self):
        '''
         Generate reservoir and thus generate the block group and pathway in it.

         Parameters
         ----------
         '''

        reservoir = Reservoir()
        structure_type = self.decoder.get_reservoir_structure_type()
        p_connection = self.decoder.get_pathway_structure('Reservoir_config')['p_connection']
        blocks_position, connection_matrix, o, i = self.generate_connection_matrix_reservoir(structure_type)
        block_group = self.generate_blocks(blocks_position)
        pathway = self.generate_pathway('pathway_reservoir_', block_group, block_group, connection_matrix,
                                        dynamics_reservoir_synapse_STDP, dynamics_reservoir_synapse_pre_STDP,
                                        dynamics_reservoir_synapse_post_STDP)
        pathway.connect(p_connection = p_connection, connect_type = 'probability')
        reservoir.register_blocks(block_group)
        reservoir.register_pathway(pathway)
        reservoir.register_input_output(o, i)
        return reservoir

    def generate_encoding(self):
        '''
         Generate a block group containing only one block as encoding layer.

         Parameters
         ----------
         '''

        block_group = BlockGroup()
        N = self.decoder.get_encoding_structure()
        block = Block(N, np.array([]).reshape(2,-1))
        block.create_neurons(dynamics_encoding, threshold='I > 0', reset = None,
                             refractory = 0 * ms , name='block_encoding')
        block.create_synapse('strength : 1', None,
                            None, name='block_block_encoding_0')
        block.determine_input_output()
        block_group.add_block(block, -1)
        return block_group

    def generate_readout(self, reservoir):
        '''
         Generate a block group containing only one block as readout layer.
         The number of neurons is based on the reservoir.

         Parameters
         ----------
         reservoir: Reservoir, a instance of Reservoir class.
         '''

        block_group = BlockGroup()
        N = reservoir.total_neurons_count
        block = Block(N, np.array([]).reshape(2,-1))
        block.create_neurons(dynamics_readout, threshold=None, reset = None,
                             refractory = False, name='block_readout')
        block.create_synapse('strength : 1', None,
                            None, name='block_block_readout_0')
        block.determine_input_output()
        block_group.add_block(block, -1)
        return block_group

    def generate_pathway_encoding_reservoir(self, encoding, reservoir):
        '''
         Generate pathway between the block group of encoding and the block group of reservoir.

         Parameters
         ----------
         encoding: BlockGroup, a instance of BlockGroup class only containing one Block as encoding layer.
         reservoir: Reservoir, a instance of Reservoir class.
         '''

        connection_matrix = [[0]*len(reservoir.input), reservoir.input]
        p_connection = self.decoder.get_pathway_structure('Encoding_Readout')['p_connection']
        pathway = self.generate_pathway('pathway_encoding_', encoding, reservoir.block_group, connection_matrix,
                                        dynamics_encoding_synapse_STDP, dynamics_encoding_synapse_pre_STDP,
                                        dynamics_encoding_synapse_post_STDP)
        pathway.connect(p_connection = p_connection, connect_type = 'probability')
        return pathway

    def generate_pathway_reservoir_readout(self, reservoir, readout):
        '''
         Generate pathway between the block group of readout and the block group of reservoir.

         Parameters
         ----------
         reservoir: Reservoir, a instance of Reservoir class.
         readout: BlockGroup, a instance of BlockGroup class only containing one Block as readout layer.
         '''

        connection_matrix = [reservoir.all, [0]*len(reservoir.all)]
        pathway = self.generate_pathway('pathway_readout_', reservoir.block_group, readout, connection_matrix,
                                        'strength = 1 : 1', dynamics_readout_synapse_pre, None)
        pathway.connect(connect_type = 'one_to_one_all')
        return pathway

    def generate_network(self):
        '''
         A comprehensive structure generation function of the LSM_Network.

         Parameters
         ----------
         '''

        network = LSM_Network()
        encoding = self.generate_encoding()
        reservoir = self.generate_reservoir()
        readout = self.generate_readout(reservoir)
        pathway_encoding_reservoir = self.generate_pathway_encoding_reservoir(encoding, reservoir)
        pathway_reservoir_readout = self.generate_pathway_reservoir_readout(reservoir, readout)

        network.register_layer(encoding, 'encoding')
        network.register_layer(reservoir, 'reservoir')
        network.register_layer(readout, 'readout')
        network.register_pathway(pathway_encoding_reservoir, 'encoding_reservoir')
        network.register_pathway(pathway_reservoir_readout, 'reservoir_readout')
        return network

    def pre_initialize_block(self, position):
        '''
         Initialize the block with the parameters form decoder.

         Parameters
         ----------
         position: int, the block order in a basic block group.
         '''

        block_type = self.decoder.get_block_type(position)
        parameters = self.decoder.get_block_parameter(position)
        parameters_neurons = self.get_sub_dict(parameters, 'tau', 'tau_I')
        parameters_neurons['v'] = voltage_reset
        parameters_neurons['threshold'] = threshold_solid
        parameters_synapses = self.get_sub_dict(parameters, 'plasticity', 'strength', 'type')
        self.change_dict_key(parameters_synapses, 'strength', 'strength_need_random')
        return parameters_neurons, parameters_synapses

    def pre_initialize_reservoir(self):
        '''
         Initialize the components of reservoir, which contains pathway and block gourp.

         Parameters
         ----------
         '''
        parameter_block_neurons = {}
        parameter_block_synapses = {}
        for position in range(self.block_types_N):
            parameter_block_neurons[position], parameter_block_synapses[position] = self.pre_initialize_block(position)
        parameter_block_group = {'parameter_block_neurons':parameter_block_neurons,
                                 'parameter_block_synapses':parameter_block_synapses}
        parameters_reservoir = {'parameter_block_group':parameter_block_group,
                                'parameter_pathway': self.decoder.get_pathway_parameter('Reservoir_config')}
        self.change_dict_key(parameters_reservoir['parameter_pathway'], 'strength', 'strength_need_random')
        return parameters_reservoir

    def pre_initialize_readout(self):
        '''
         Initialize the components of readout, readout is a block group.

         Parameters
         ----------
         '''

        return  {'parameter_block_neurons': {-1: self.decoder.get_readout_parameter()}}

    def pre_initialize_encoding_reservoir(self):
        '''
         Initialize the components of encoding, encoding is a block group.

         Parameters
         ----------
         '''

        parameters_encoding_reservoir = self.decoder.get_pathway_parameter('Encoding_Readout')
        self.decoder.change_dict_key(parameters_encoding_reservoir, 'strength', 'strength_need_random')
        return parameters_encoding_reservoir

    def pre_initialize_network(self):
        '''
         Initialize the components of network.

         Parameters
         ----------
         '''

        parameters = {}
        parameters['encoding'] = None
        parameters['reservoir'] = self.pre_initialize_reservoir()
        parameters['readout'] = self.pre_initialize_readout()
        parameters['encoding_reservoir'] = self.pre_initialize_encoding_reservoir()
        parameters['reservoir_readout'] = None

        return parameters


    def initialize(self, network):
        '''
         Initialize the components of LSM_Network,
         it will call the function 'initialize' of LSM_Network.

         Parameters
         ----------
         network: LSM_Network, the instance of 'LSM_Network'.
         '''

        network.initialize(**self.pre_initialize_network())

    def join(self, net, network):
        '''
         Join the network generated by generator into the 'Brain2.Network'.
         Only used after the LSM_Network generated,
         it will call the function 'join_network' of LSM_Network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         network: LSM_Network, the instance of 'LSM_Network'.
         '''

        network.join_network(net)