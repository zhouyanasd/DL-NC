# -*- coding: utf-8 -*-
"""
    The fundamental neurons and network structure
    including local blocks.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from .core import BaseFunctions

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

    def separate_ex_inh(self, random_state = None):
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

    # def initialize_parameters(self, object, parameter_name, parameter_value):
    #     '''
    #      Set the initial parameters of the objects in the block.
    #
    #      Parameters
    #      ----------
    #      object: Brian2.NeuronGroup or Brian2.Synapse, one of the two kinds of objects.
    #      parameter_name: str, the name of the parameter.
    #      parameter_value: np.array, the value of the parameter.
    #      '''
    #     if isinstance(object, NeuronGroup):
    #         object.variables[parameter_name].set_value(parameter_value)
    #     elif isinstance(object, Synapses):
    #         object.pre.variables[parameter_name].set_value(parameter_value)
    #     else:
    #         print('wrong object type')

    def join_network(self, net):
        '''
         Let the objects of block join the whole neural network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         '''
        net.add(self.neurons, self.synapse)

    def determine_input_output(self, blocks_input, blocks_output):
        '''
         Determine the index of input and output neurons.

         Parameters
         ----------
         '''
        self.input = [1, 2]
        self.output = [8, 9]


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
         Create neurons group for the block.

         Parameters
         ----------
         The parameters follow the necessary 'NeuronGroup' class of Brain2.
         '''

        self.blocks.append(block)

    def create_blocks(self, neurons_block, connect_matrix_blocks, dynamics_neurons, dynamics_synapse,
                      dynamics_synapse_pre, threshold, reset, refractory):

        for index, connect_matrix in zip(range(self.N), connect_matrix_blocks):
            block_reservoir = Block(neurons_block)
            block_reservoir.create_neurons(dynamics_neurons, threshold = threshold, reset = reset,
                                           refractory = refractory, name='block_' + str(index))
            block_reservoir.create_synapse(dynamics_synapse, dynamics_synapse_pre,
                                           name='block_block_' + str(index))
            block_reservoir.connect(connect_matrix)
            self.add_block(block_reservoir)

    def create_synapse(self, connect_matrix, model, on_pre, blocks_input, blocks_output, **kwargs):
        '''
         Create synapses between blocks.

         Parameters
         ----------
         connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse between blocks.
                        The first list is the pre-synapse blocks and the second list is the post-synapse blocks.
         Other parameters follow the necessary 'Synapses' class of Brain2.
         '''

        for block in self.blocks:
            block.determine_input_output(blocks_input, blocks_output)
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
        for block_reservoir in self.blocks:
            for key, value in zip(kwargs.keys(), kwargs.values()):
                block_reservoir.initialize_parameters(block_reservoir.neurons, key, value)

    def initialize_blocks_synapses(self, connect_matrix_blocks, **kwargs):
        for index, block_reservoir in enumerate(self.blocks):
            for key, value in zip(kwargs.keys(), kwargs.values()):
                converted_value = self.get_parameters(connect_matrix_blocks[index], value)
                block_reservoir.initialize_parameters(block_reservoir.synapse, key, converted_value)

    def initialize_reservoir_synapses(self, **kwargs):
        for index, synapse in enumerate(self.synapses):
            for key, value in zip(kwargs.keys(), kwargs.values()):
                block_pre_index = self.connect_matrix[0][index]
                block_post_index = self.connect_matrix[1][index]
                block_pre = self.blocks[block_pre_index]
                block_post = self.blocks[block_post_index]
                connect_matrix = self.np_two_combination(block_pre.output, block_post.input)

                parameter_synapse = np.zeros(block_pre.N, block_post.N)
                parameter = value[block_pre_index][block_post_index]
                parameter_ = np.random.rand(len(block_pre.output) * len(block_post.input)) * parameter
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


class LSM_Network(Network):
    def __init__(self):
        super().__init__()
        self.layers = {}
        self.synapses_readout = []
        self.synapses_encoding = []

    def register_layer(self, layer, name):
        self.layers[name] = layer
        try:
            self.add(layer)
        except TypeError:
            layer.join_network(self)

    def create_synapse_encoding(self, dynamics_synapse, dynamics_synapse_pre, reservoir_input):
        self.layers['reservoir'].determine_input(reservoir_input)
        for index, block_reservoir in enumerate(self.layers['reservoir'].input):
            synapse_encoding_reservoir = Synapses(self.layers['encoding'], block_reservoir.neurons, dynamics_synapse,
                                                  on_pre=dynamics_synapse_pre,
                                                  method='euler', name='encoding_block_' + str(index))
            self.synapses_encoding.append(synapse_encoding_reservoir)

    def create_synapse_readout(self, dynamics_synapse_pre):
        for index, block_reservoir in enumerate(self.layers['reservoir']):
            synapse_reservoir_readout = Synapses(block_reservoir.neurons, self.layers['readout'], 'w = 1 : 1',
                                                 on_pre = dynamics_synapse_pre, name='readout_' + str(index))
            self.synapses_readout.append(synapse_reservoir_readout)

    def connect_encoding(self):
        for index, synapse_encoding_reservoir in enumerate(self.synapses_encoding):
            block_post = self.layers['reservoir'].input[index]
            input = block_post.input
            synapse_encoding_reservoir.connect(j='k for k in input') # var 'input' needs to be test

    def connect_readout(self):
        neurons_block = self.layers['reservoir'].blocks[0].N
        for index, synapse_reservoir_readout in enumerate(self.synapses_readout):
            synapse_reservoir_readout.connect(j='i+' + str(index * neurons_block))

    def initialize_parameters_encoding_synapses(self, **kwargs):
        pass

    def join_network(self):
        self.add(*self.synapses_readout, *self.synapses_encoding)