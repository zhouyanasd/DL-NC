# -*- coding: utf-8 -*-
"""
    The components generator based on the parameter
    decoded from optimizer.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""


from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.networks.components import *
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.networks.decoder import *

from brian2 import *


class Generator():
    '''
     Initialize the parameters of the neurons or synapses.

     Parameters
     ----------
     # some public used such as random_state.
     '''

    def __init__(self, random_state):
        self.random_state = random_state

    def generate_block(self):
        pass

    def generate_pathway(self):
        pass

    def generate_parameters(self):
        pass

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