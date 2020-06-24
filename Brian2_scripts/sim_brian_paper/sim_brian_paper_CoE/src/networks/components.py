# -*- coding: utf-8 -*-
"""
    The fundamental neurons and network structure
    including local blocks.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import Topological_sorting_tarjan

from brian2 import *

class Block(BaseFunctions):
    """
    This class offers a basic property and functions of block in LSM.

    Parameters
    ----------
    N: int, the number of neurons
    ratio: default int(4), the ratio of excitatory neurons/inhibitory neurons
    connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse.
    """

    def __init__(self, N, ratio = 4):
        super().__init__()
        self.N = N
        self.ex_inh_ratio = ratio
        self.connect_matrix = None
        self.input = None
        self.output = None

    def separate_ex_inh(self, random_state = None):
        '''
         Generate neural property to separate the ex/inh neurons.

         Parameters
         ----------
         random_state: int, the random seed needs to be set when np.shuffle the property.
         '''

        self.ex_neurons = int(self.N * (1 / (self.ex_inh_ratio + 1)))
        self.inh_neurons = self.N - self.ex_neurons
        self.neuron_property = np.array(([-1] * self.ex_neurons) + ([1] * self.inh_neurons))
        if random_state != None:
            np.random.seed(random_state)
        np.random.shuffle(self.neuron_property)

    def create_neurons(self, model, threshold, reset,refractory, name, **kwargs):
        '''
         Create neurons group for the block.

         Parameters
         ----------
         The parameters follow the necessary 'NeuronGroup' class of Brain2.
         '''
        self.neurons = NeuronGroup(self.N, model, threshold=threshold, reset=reset, refractory=refractory,
                                   method='euler', name = name, **kwargs)
        try:
            self.separate_ex_inh(random_state = kwargs['random_state'])
        except KeyError:
            self.separate_ex_inh()
        self.initialize_parameters(self.neurons, 'property', self.neuron_property)

    def create_synapse(self, model, on_pre, name, **kwargs):
        '''
         Create synapse between neurons for the block.

         Parameters
         ----------
         The parameters follow the necessary 'Synapses' class of Brain2.
         '''
        self.synapse = Synapses(self.neurons, self.neurons, model, on_pre = on_pre,
                                method='euler', name = name, **kwargs)

    def connect(self, connect_matrix):
        '''
         Connect neurons using synapse based on the fixed connection matrix.

         Parameters
         ----------
        connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse.
                        The first list is the pre-synapse neurons and the second list is the post-synapse neurons.
         '''
        self.connect_matrix = connect_matrix
        self.synapse.connect(i = connect_matrix[0], j = connect_matrix[1])

    def join_network(self, net):
        '''
         Let the objects of block join the whole neural network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         '''
        net.add(self.neurons, self.synapse)

    def determine_input_output(self):
        '''
         Determine the index of input and output neurons.
         The input and output are list, e.g. [1,2], [3,4].
         '''
        adjacent_matrix = self.connection_matrix_to_adjacent_matrix(self.connect_matrix)
        topological_sorting_tarjan = Topological_sorting_tarjan(adjacent_matrix)
        topological_sorting_tarjan.dfs()
        self.input, self.output = topological_sorting_tarjan.suggest_inout()


class Pipeline(BaseFunctions):
    '''
     Pipeline contains the synapses between blocks, which
     offer the basic synaptic-like function of connection.

     Parameters
     ----------
     '''

    def __init__(self, connect_matrix, model, on_pre, on_post,  name = name, **kwargs):
        super().__init__()
        self.pre = connect_matrix[0]
        self.post = connect_matrix[1]
        self.synapse = []

    def connect(self):
        pass

class Reservoir(BaseFunctions):
    """
    This class offers a basic property and functions of reservoir containing blocks.

    Parameters
    ----------
    N: int, the number of blocks
    connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse.
    blocks: list[Block], the list of block
    synapses: list[Brain.Synapse], the list of synapse between blocks
    """

    def __init__(self, N):
        super().__init__()
        self.N = N
        self.connect_matrix = None
        self.blocks = []
        self.synapses = []

    def add_block(self, block):
        '''
         Add block to the reservoir.

         Parameters
         ----------
         block: Block, the object of Block.
         '''

        self.blocks.append(block)

    def create_blocks(self, neurons_block, connect_matrix_blocks, dynamics_neurons, dynamics_synapse,
                      dynamics_synapse_pre, threshold, reset, refractory):
        '''
         Create blocks for the reservoir.

         Parameters
         ----------
         neurons_block: int, the number of neurons in each block.
         connect_matrix_blocks: list, a list of connect_matrix for the block.
         Other parameters follow the necessary 'Synapses' class of Brain2.
         '''

        for index, connect_matrix in zip(range(self.N), connect_matrix_blocks):
            block_reservoir = Block(neurons_block)
            block_reservoir.create_neurons(dynamics_neurons, threshold = threshold, reset = reset,
                                           refractory = refractory, name='block_' + str(index))
            block_reservoir.create_synapse(dynamics_synapse, dynamics_synapse_pre,
                                           name='block_block_' + str(index))
            block_reservoir.connect(connect_matrix)
            self.add_block(block_reservoir)

    def create_synapse(self, connect_matrix, model, on_pre, **kwargs):
        '''
         Create synapses between blocks.

         Parameters
         ----------
         connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse between blocks.
                        The first list is the pre-synapse blocks and the second list is the post-synapse blocks.
         Other parameters follow the necessary 'Synapses' class of Brain2.
         '''

        for block in self.blocks:
            block.determine_input_output()
        for index, (index_i, index_j) in enumerate(zip(connect_matrix[0], connect_matrix[1])):
            synapse = Synapses(block[index_i].neurons, block[index_j].neurons, model, on_pre = on_pre,
                                method = 'euler', name = 'reservoir_' + str(index), **kwargs)
            self.synapses.append(synapse)

    def connect_blocks(self):
        '''
         Connect blocks base on the synapses in the reservoir.

         Parameters
         ----------
         '''

        for index, synapse in enumerate(self.synapses):
            block_pre = self.blocks[self.connect_matrix[0][index]]
            block_post = self.blocks[self.connect_matrix[1][index]]
            connect_matrix = self.np_two_combination(block_pre.output, block_post.input)
            synapse.connect(i = connect_matrix[0], j = connect_matrix[1])

    def initialize_blocks_neurons(self, **kwargs):
        '''
         Initialize the parameters of the neurons of blocks in the reservoir.

         Parameters
         ----------
         kwargs: dict{key:str(name), value:np.array[n*1]}, extensible dict of parameters.
         '''

        for block_reservoir in self.blocks:
            for key, value in zip(kwargs.keys(), kwargs.values()):
                block_reservoir.initialize_parameters(block_reservoir.neurons, key, value)

    def initialize_blocks_synapses(self, **kwargs):
        '''
         Initialize the parameters of the synapses of blocks in the reservoir.

         Parameters
         ----------
         kwargs: dict{key:str(name), value:np.array[n*n]}, extensible dict of parameters.
         '''

        for block_reservoir in self.blocks:
            for key, value in zip(kwargs.keys(), kwargs.values()):
                converted_value = self.get_parameters(block_reservoir.connect_matrix, value)
                block_reservoir.initialize_parameters(block_reservoir.synapse, key, converted_value)

    def initialize_reservoir_synapses(self, **kwargs):
        '''
         Initialize the parameters of the synapses between blocks in the reservoir.

         Parameters
         ----------
         kwargs: dict{key:str(name), value:np.array[n*n]}, extensible dict of parameters.
         '''

        for index, synapse in enumerate(self.synapses):
            for key, value in zip(kwargs.keys(), kwargs.values()):
                connect_matrix = [list(synapse.i), list(synapse.j)]

                parameter_synapse = np.zeros(synapse.N_pre[0], synapse.N_post[0])
                parameter = value[self.output[index]][self.input[index]]
                parameter_ = np.random.rand(synapse.N[0] * synapse.N[0]) * parameter
                for index, (index_i, index_j) in enumerate(zip(connect_matrix)):
                    parameter_synapse[index_i][index_j] = parameter_[index]

                converted_value = self.get_parameters(connect_matrix, parameter_synapse)
                self.initialize_parameters(synapse, key, converted_value)

    def join_network(self, net):
        '''
         Let the objects of reservoir join the whole neural network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         '''

        for block in self.blocks:
            block.join_network(net)
        net.add(*self.synapses)

    def determine_input(self, reservoir_input):
        '''
         Determine the index of input and output neurons.

         Parameters
         ----------
         '''
        self.input = [1, 2]


class LSM_Network(Network, BaseFunctions):
    """
    This class offers a basic property and functions of network
    containing reservoir, encoding and readout.

    Parameters
    ----------
    layers: dict, contained layers.
    synapses_encoding: list[Brain.Synapse], the list of synapse between encoding and blocks.
    synapses_readout: list[Brain.Synapse], the list of synapse between readout and blocks.
    """

    def __init__(self):
        super().__init__()
        self.layers = {}
        self.synapses_readout = []
        self.synapses_encoding = []

    def register_layer(self, layer, name):
        '''
         Register layers in the network.

         Parameters
         ----------
         layer: Reservoir, Brain2.NeuronGroup, the layer in network.
         name: str, the name for the dict.
         '''

        self.layers[name] = layer
        try:
            self.add(layer)
        except TypeError:
            layer.join_network(self)

    def create_synapse_encoding(self, dynamics_synapse, dynamics_synapse_pre, reservoir_input):
        '''
         Create synapses between encoding and blocks.

         Parameters
         ----------
         reservoir_input: int, the number of blocks in the reservoir used for input.
         Other parameters follow the necessary 'Synapses' class of Brain2.
         '''

        self.layers['reservoir'].determine_input(reservoir_input)
        for index, block_reservoir_index in enumerate(self.layers['reservoir'].input):
            synapse_encoding_reservoir = Synapses(self.layers['encoding'],
                                                  self.layers['reservoir'].blocks[block_reservoir_index].neurons,
                                                  dynamics_synapse,
                                                  on_pre=dynamics_synapse_pre,
                                                  method='euler', name='encoding_block_' + str(index))
            self.synapses_encoding.append(synapse_encoding_reservoir)

    def create_synapse_readout(self, dynamics_synapse_pre):
        '''
         Create synapses between readout and blocks.

         Parameters
         ----------
         Other parameters follow the necessary 'Synapses' class of Brain2.
         '''

        for index, block_reservoir in enumerate(self.layers['reservoir'].blocks):
            synapse_reservoir_readout = Synapses(block_reservoir.neurons, self.layers['readout'], 'w = 1 : 1',
                                                 on_pre = dynamics_synapse_pre, name='readout_' + str(index))
            self.synapses_readout.append(synapse_reservoir_readout)

    def connect_encoding(self):
        '''
         Connect blocks and encoding base on the synapses in the network.
         The connection is full connection from the encoding layer to the input blocks.

         Parameters
         ----------
         '''

        for index, synapse_encoding_reservoir in enumerate(self.synapses_encoding):
            block_post_index = self.layers['reservoir'].input[index]
            block_post = self.layers['reservoir'].blocks[block_post_index]
            input = block_post.input
            output = np.arange(self.layers['encoding'].N[0])
            connect_matrix = self.np_two_combination(output, input)
            synapse_encoding_reservoir.connect(i = connect_matrix[0], j = connect_matrix[1]) # var 'input' needs to be test

    def connect_readout(self):
        '''
         Connect blocks and readout base on the synapses in the network.
         The connections is one to one.

         Parameters
         ----------
         '''

        neurons_block = self.layers['reservoir'].blocks[0].N
        for index, synapse_reservoir_readout in enumerate(self.synapses_readout):
            synapse_reservoir_readout.connect(j='i+' + str(index * neurons_block))

    def initialize_parameters_encoding_synapses(self, **kwargs):
        '''
         Initialize the parameters of the synapses between reservoir and encoding.

         Parameters
         ----------
         kwargs: dict{key:str(name), value:np.array[n*1]}, extensible dict of parameters.
         '''
        for index, synapse in enumerate(self.synapses_encoding):
            for key, value in zip(kwargs.keys(), kwargs.values()):
                connect_matrix = [list(synapse.i), list(synapse.j)]

                parameter_synapse = np.zeros(synapse.N_pre[0], synapse.N_post[0])
                parameter = value[self.layers['reservoir'].input[index]]
                parameter_ = np.random.rand(synapse.N[0] * synapse.N[0]) * parameter
                for index, (index_i, index_j) in enumerate(zip(connect_matrix)):
                    parameter_synapse[index_i][index_j] = parameter_[index]

                converted_value = self.get_parameters(connect_matrix, parameter_synapse)
                self.initialize_parameters(synapse, key, converted_value)

    def join_network(self):
        '''
         Let the synapses of encoding and readout join the whole neural network.

         Parameters
         ----------
         '''

        self.add(*self.synapses_readout, *self.synapses_encoding)