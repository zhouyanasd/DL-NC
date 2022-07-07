# -*- coding: utf-8 -*-
"""
    The components generator based on the parameter
    decoded from optimizer.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""


from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src.networks.components import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src.config import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src.core import *

from brian2 import *


def convert_networkx(G):
    N = G.number_of_nodes()
    edges = G.edges()
    connection_matrix_out, connection_matrix_in = [e[0] for e in edges], [e[1] for e in edges]
    return N, np.array([connection_matrix_out, connection_matrix_in])


class Generator_connection_matrix(NetworkBase):
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
        # self.block_types_N = len(list(structure_blocks))

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
        elif component['name'] == 'small_world_2':
            parameters_['p_forward'] = self.adapt_scale(component['p_0'], parameters['p_0'])
            parameters_['p_backward'] = self.adapt_scale(component['p_1'], parameters['p_1'])
            parameters_['p_threshold'] = self.adapt_scale(component['p_2'], parameters['p_2'])
        elif component['name'] == 'three_layer':
            parameters_['p_out'] = self.adapt_scale(component['p_0'], parameters['p_0'])
            parameters_['p_in'] = self.adapt_scale(component['p_1'], parameters['p_1'])
            parameters_['decay'] = self.adapt_scale(component['p_2'], parameters['p_2'])

        block_generator_type = {'random': self.generate_connection_matrix_blocks_random,
                                'scale_free': self.generate_connection_matrix_blocks_scale_free,
                                'small_world_2': self.generate_connection_matrix_blocks_small_world_2,
                                'three_layer': self.generate_connection_matrix_blocks_three_layer}
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

        alpha = p_alpha / (p_alpha + p_beta + p_gama)
        beta = p_beta / (p_alpha + p_beta + p_gama)
        gama = p_gama / (p_alpha + p_beta + p_gama)
        DSF = Direct_scale_free(final_nodes=N, alpha=alpha, beta=beta, gama=gama,
                                init_nodes=1, delta_in=0, delta_out=0)
        DSF.generate_graph()
        connection_matrix_out, connection_matrix_in = DSF.o, DSF.i
        return N, np.array([connection_matrix_out, connection_matrix_in])

    def generate_connection_matrix_blocks_small_world(self, N, p_k, p_reconnect):
        '''
         Generate connection matrix of small world.

         Parameters
         ----------
         N: int, number of neurons of the block.
         p_k: float, each node is joined with its `np.ceil(p_k*N)` nearest neighbors in a ring topology.
         p_reconnect： float, the probability of rewriting a new edge for each edge.，

         See Also
         --------
         watts_strogatz_graph

         References
         ----------
         [1] Duncan J. Watts and Steven H. Strogatz,
             Collective dynamics of small-world networks,
             Nature, 393, pp. 440--442, 1998.
         [2] https://github.com/networkx/networkx/blob/main/networkx/generators/random_graphs.py
         '''

        if p_k >=1:
            return self.generate_connection_matrix_blocks_random(N, 1)

        k = np.floor(p_k * N).astype(np.int)
        k_h = np.floor(0.5 * k).astype(np.int)

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
                if distance <= k_h:
                    connection_matrix_out.append(node_pre)
                    connection_matrix_in.append(node_post)
                    connection_matrix_out.append(node_post)
                    connection_matrix_in.append(node_pre)
                else:
                    break

        edges = list(zip(connection_matrix_out, connection_matrix_in))

        for index_pre, node_pre in enumerate(circle[:N]):
            for index_post, node_post in enumerate(circle[index_pre+1:N+index_pre+1]):
                distance = index_post
                if distance <= k_h and np.random.rand() <= p_reconnect:
                    node = np.random.choice(nodes)
                    while node_pre == node or (node, node_post) in edges:
                        node = np.random.choice(nodes)
                        if sum(np.array(connection_matrix_out) == node) >= N-1:
                            break
                    else:
                        edges.remove((node_pre, node_post))
                        edges.append((node_pre, node))

        connection_matrix_out, connection_matrix_in = [e[0] for e in edges], [e[1] for e in edges]

        return N, np.array([connection_matrix_out, connection_matrix_in])

    def generate_connection_matrix_blocks_small_world_2(self, N, p_forward, p_backward, p_threshold):
        '''
         Generate connection matrix of direct small_world block.
         This method is generated based on a 'circle' structure.
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


class Generator(BaseFunctions):
    """
    This class offers a basic generation functions of each component in network.
    **This class can not be instantiated, use subclass instead.**

    Parameters
    ----------
    random_state: int, random state generated by np.random.
    """

    def __init__(self, random_state):
        self.random_state = random_state

    def register_decoder(self, decoder):
        '''
         Add gen decoder to this generator.

         Parameters
         ----------
         decoder: Decoder, a instance of Decoder class.
         '''

        self.decoder = decoder

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


class Generator_Block(Generator, Generator_connection_matrix):
    """
    This subclass offers the generation functions of each component for the block.

    Parameters
    ----------
    random_state: int, random state generated by np.random.
    """

    def __init__(self, random_state, task_id):
        super().__init__(random_state)
        self.task_id = task_id

    def generate_block(self, index):
        '''
         A basic block generator function.

         Parameters
         ----------
         index: int, the block index of all blocks in reservoir.
         '''

        block_type =  self.decoder.get_block_type()
        parameter_structure = self.decoder.get_block_structure()
        component_name = structure_blocks['components_' + str(block_type)]['name'] + '_' + str(index)
        N, connect_matrix = self.generate_connection_matrix_blocks(block_type, **parameter_structure)
        block = Block(N, connect_matrix)
        block.create_neurons(dynamics_reservoir, threshold = threshold_reservoir, reset = reset_reservoir,
                             refractory = refractory_reservoir, name='block_' + component_name+'_task'+str(self.task_id))
        block.create_synapse(dynamics_block_synapse_STDP, dynamics_block_synapse_pre_STDP,
                             dynamics_block_synapse_post_STDP, name='block_block_' + component_name+'_task'+str(self.task_id))
        block.separate_ex_inh()
        block.connect()
        block.determine_input_output()
        return block

    def generate_reservoir_block_single(self):
        '''
         Generate a block group containing only one block as encoding layer.

         Parameters
         ----------
         '''

        reservoir = Reservoir()
        block_group = BlockGroup()
        block = self.generate_block(index=0)
        block_group.add_block(block)
        pathway = Pathway(block_group.blocks, block_group.blocks, [[],[]])
        pathway.create_synapse(dynamics_reservoir_synapse_STDP, dynamics_reservoir_synapse_pre_STDP,
                               dynamics_reservoir_synapse_post_STDP,
                               name = 'pathway_block_single_task'+str(self.task_id)+'_')
        pathway.connect(p_connection= 0, connect_type='probability')
        reservoir.register_blocks(block_group)
        reservoir.register_pathway(pathway)
        reservoir.register_input_output(o=[0], i=[0])
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
                             refractory = 0 * ms , name='block_encoding_task'+str(self.task_id))
        block.create_synapse('strength : 1', None, None, name='block_block_encoding_task'+str(self.task_id)+'_0')
        block.determine_input_output()
        block_group.add_block(block, self.task_id)
        return block_group

    def generate_readout(self, reservoir):
        '''
         Generate a block group containing only one block as readout layer.
         The number of neurons is based on the reservoir.

         Parameters
         ----------
         reservoir: Reservoir, an instance of Reservoir class.
         '''

        block_group = BlockGroup()
        N = reservoir.total_neurons_count
        block = Block(N, np.array([]).reshape(2,-1))
        block.create_neurons(dynamics_readout, threshold=None, reset = None,
                             refractory = False, name='block_readout_task'+str(self.task_id))
        block.create_synapse('strength : 1', None, None, name='block_block_readout_task'+str(self.task_id)+'_0')
        block.determine_input_output()
        block_group.add_block(block, self.task_id)
        return block_group

    def generate_pathway_encoding_reservoir(self, encoding, reservoir):
        '''
         Generate pathway between the block group of encoding and the block group of reservoir.

         Parameters
         ----------
         encoding: BlockGroup, an instance of BlockGroup class only containing Blocks for each task as encoding layer.
         reservoir: Reservoir, an instance of Reservoir class.
         '''

        connection_matrix = [[0]*len(reservoir.input), reservoir.input]
        p_connection = self.decoder.get_pathway_structure('Encoding_Readout')['p_connection']
        pathway = Pathway(encoding.blocks, reservoir.block_group.blocks, connection_matrix)
        pathway.create_synapse(dynamics_encoding_synapse_STDP, dynamics_encoding_synapse_pre_STDP,
                               dynamics_encoding_synapse_post_STDP,
                               name = 'pathway_encoding_task'+str(self.task_id)+'_')
        pathway.connect(p_connection = p_connection, connect_type = 'probability')
        return pathway

    def generate_pathway_reservoir_readout(self, reservoir, readout):
        '''
         Generate pathway between the block group of readout and the block group of reservoir.

         Parameters
         ----------
         reservoir: Reservoir, an instance of Reservoir class.
         readout: BlockGroup, an instance of BlockGroup class only containing one Block as readout layer.
         '''

        connection_matrix = [reservoir.all, [0]*len(reservoir.all)]
        pathway = Pathway(reservoir.block_group.blocks, readout.blocks, connection_matrix)
        pathway.create_synapse('strength = 1 : 1', dynamics_readout_synapse_pre, None,
                               name = 'pathway_readout_task'+str(self.task_id)+'_')
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
        reservoir = self.generate_reservoir_block_single()
        readout = self.generate_readout(reservoir)
        pathway_encoding_reservoir = self.generate_pathway_encoding_reservoir(encoding, reservoir)
        pathway_reservoir_readout = self.generate_pathway_reservoir_readout(reservoir, readout)

        network.register_layer(encoding, 'encoding')
        network.register_layer(reservoir, 'reservoir')
        network.register_layer(readout, 'readout')

        network.register_pathway(pathway_encoding_reservoir, 'encoding_reservoir_task'+str(self.task_id))
        network.register_pathway(pathway_reservoir_readout, 'reservoir_readout_task'+str(self.task_id))
        return network

    def pre_initialize_block(self):
        '''
         Initialize the block with the parameters form decoder.

         Parameters
         ----------
         '''

        parameters = self.decoder.get_block_parameter(self.task_id)
        parameters_neurons = self.get_sub_dict(parameters, 'tau', 'tau_I')
        parameters_neurons['v'] = voltage_reset
        parameters_neurons['threshold'] = threshold_solid
        parameters_synapses = self.get_sub_dict(parameters, 'tau_plasticity', 'strength', 'type')
        self.change_dict_key(parameters_synapses, 'strength', 'strength_need_random')
        return parameters_neurons, parameters_synapses

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

    def pre_initialize_reservoir_single_block(self):
        '''
         Initialize the components of reservoir, which contains pathway and block group.

         Parameters
         ----------
         '''

        parameter_block_neurons = {}
        parameter_block_synapses = {}
        parameter_block_neurons[self.task_id], parameter_block_synapses[self.task_id] = self.pre_initialize_block()
        parameter_block_group = {'parameter_block_neurons':parameter_block_neurons,
                                 'parameter_block_synapses':parameter_block_synapses}
        parameters_reservoir = {'parameter_block_group':parameter_block_group,
                                'parameter_pathway': {'type': 0.0, 'tau_plasticity': 0.0, 'strength': 0.0}}
        self.change_dict_key(parameters_reservoir['parameter_pathway'], 'strength', 'strength_need_random')
        return parameters_reservoir

    def pre_initialize_network(self):
        '''
         Initialize the components of network.

         Parameters
         ----------
         '''

        parameters = {}
        parameters['encoding'] = None
        parameters['reservoir'] = self.pre_initialize_reservoir_single_block()
        parameters['readout'] = self.pre_initialize_readout()
        parameters['encoding_reservoir'] = self.pre_initialize_encoding_reservoir()
        parameters['reservoir_readout'] = None

        return parameters

class Generator_Reservoir(Generator):
    """
    This subclass offers the generation functions of each component for the reservoir.

    Parameters
    ----------
    random_state: int, random state generated by np.random.
    """

    def __init__(self, random_state, block_init, block_max):
        super().__init__(random_state)
        self.block_generators = {}
        self.block_init = block_init
        self.block_max = block_max

    def register_block_generator(self, **neurons_encoding):
        '''
         Add gen decoder to this generator.

         Parameters
         ----------
         '''

        for task_id, optimal_block_gen in self.decoder.get_optimal_block_gens().items():
            block_decoder = (config_group_block, config_keys_block, config_SubCom_block,
                             config_codes_block, config_ranges_block, config_borders_block,
                             config_precisions_block, config_scales_block,
                              gen_group_block, neurons_encoding[task_id])
            block_generator = Generator_Block(self.random_state, task_id)
            block_generator.register_decoder(block_decoder)
            self.block_generators[task_id] = block_generator

    def initialize_task_ids(self):
        '''
         Generate initial task_ids for generating blocks.

         Parameters
         ----------
         '''

        self.tasks_ids = (list(self.decoder.get_optimal_block_gens().items())*\
                            np.ceil(self.block_max/self.block_init).astype(int))[:self.block_init]

    def increase_block_reservoir(self, task_id):
        '''
         Increase block for new task.

         Parameters
         ----------
         '''

        if len(self.tasks_ids) <= self.block_max:
            self.tasks_ids.append(task_id)
            block_current = len(self.tasks_ids)
            self.decoder.increase_block_reservoir(block_current, block_max)

    def generate_blocks(self):
        '''
         Generate blocks for reservoir according to blocks_type,
         the blocks belong to one block group.

         Parameters
         ----------
         '''

        block_group = BlockGroup()
        for index, task_id in enumerate(self.tasks_ids):
            block = self.block_generators[task_id].generate_block(index)
            block_group.add_block(block, task_id)
        return block_group

    def generate_reservoir(self):
        '''
         Generate reservoir and thus generate the block group and pathway in it.

         Parameters
         ----------
         '''

        reservoir = Reservoir()
        p_connection = self.decoder.get_pathway_structure('Reservoir_config')['p_connection']
        adjacent_matrix = self.decoder.get_reservoir_adjacent_matrix()
        connection_matrix = self.adjacent_matrix_to_connection_matrix(adjacent_matrix)
        topological_sorting_tarjan = Topological_sorting_tarjan(adjacent_matrix)
        topological_sorting_tarjan.dfs()
        o, i = topological_sorting_tarjan.suggest_inout()
        block_group = self.generate_blocks()
        pathway = Pathway(block_group.blocks, block_group.blocks, connection_matrix)
        pathway.create_synapse(dynamics_reservoir_synapse_STDP, dynamics_reservoir_synapse_pre_STDP,
                               dynamics_reservoir_synapse_post_STDP,  name = 'pathway_reservoir_')
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

        block_groups = []
        for generator in self.block_generators.values():
            block_groups.append(generator.generate_encoding())
        return block_groups

    def generate_readout(self, reservoir):
        '''
         Generate a block group containing only one block as readout layer.
         The number of neurons is based on the reservoir.

         Parameters
         ----------
         reservoir: Reservoir, an instance of Reservoir class.
         '''

        block_groups = []
        for generator in self.block_generators.values():
            block_groups.append(generator.generate_readout(reservoir))
        return block_groups

    def generate_pathway_encoding_reservoir(self, encodings, reservoir):
        '''
         Generate pathway between the block group of encoding and the block group of reservoir.

         Parameters
         ----------
         encodings: list[BlockGroup], all instance of BlockGroup class only containing Blocks for each task as encoding layer.
         reservoir: Reservoir, an instance of Reservoir class.
         '''

        pathways = []
        for generator, encoding in zip(self.block_generators.values(), encodings):
            pathways.append(generator.generate_pathway_encoding_reservoir(encoding, reservoir))
        return pathways

    def generate_pathway_reservoir_readout(self, reservoir, readouts):
        '''
         Generate pathway between the block group of readout and the block group of reservoir.

         Parameters
         ----------
         reservoir: Reservoir, an instance of Reservoir class.
         readouts: list[BlockGroup], all instance of BlockGroup class only containing one Block as readout layer.
         '''

        pathways = []
        for generator, readout in zip(self.block_generators.values(), readouts):
            pathways.append(generator.generate_pathway_reservoir_readout(reservoir, readout))
        return pathways

    def generate_network(self):
        '''
         A comprehensive structure generation function of the LSM_Network.

         Parameters
         ----------
         '''

        network = LSM_Network()
        reservoir = self.generate_reservoir()
        encodings = self.generate_encoding()
        readouts = self.generate_readout(reservoir)
        pathways_encoding_reservoir = self.generate_pathway_encoding_reservoir(encodings, reservoir)
        pathways_reservoir_readout = self.generate_pathway_reservoir_readout(reservoir, readouts)

        network.register_layer(reservoir, 'reservoir')
        for task_id, encoding, readout in enumerate(zip(encodings, readouts)):
            network.register_layer(encoding, 'encoding_task'+str(task_id))
            network.register_layer(readout, 'readout_task'+str(task_id))
        for task_id, pathway_encoding_reservoir, pathway_reservoir_readout \
                in enumerate(zip(pathways_encoding_reservoir, pathways_reservoir_readout)):
            network.register_pathway(pathway_encoding_reservoir, 'encoding_reservoir_task'+str(task_id))
            network.register_pathway(pathway_reservoir_readout, 'reservoir_readout_task'+str(task_id))
        return network

    def pre_initialize_reservoir(self):
        '''
         Initialize the components of reservoir, which contains pathway and block group.

         Parameters
         ----------
         '''

        parameter_block_neurons = {}
        parameter_block_synapses = {}
        for task_id, generator in self.block_generators.items():
            parameter_block_neurons[task_id], parameter_block_synapses[task_id] = generator.pre_initialize_block()
        parameter_block_group = {'parameter_block_neurons':parameter_block_neurons,
                                 'parameter_block_synapses':parameter_block_synapses}
        parameters_reservoir = {'parameter_block_group':parameter_block_group,
                                'parameter_pathway': self.decoder.get_pathway_parameter('Reservoir_config')}
        self.change_dict_key(parameters_reservoir['parameter_pathway'], 'strength', 'strength_need_random')
        return parameters_reservoir

    def pre_initialize_network(self):
        '''
         Initialize the components of network.

         Parameters
         ----------
         '''

        parameters = {}
        parameters['encoding'] = None
        parameters['reservoir'] = self.pre_initialize_reservoir()
        for task_id, generator in self.block_generators.items():
            parameters['readout_task'+str(task_id)] = generator.pre_initialize_readout()
            parameters['encoding_reservoir_task'+str(task_id)] = generator.pre_initialize_encoding_reservoir()
        parameters['reservoir_readout'] = None

        return parameters
