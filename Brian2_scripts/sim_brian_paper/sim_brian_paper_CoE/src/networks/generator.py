# -*- coding: utf-8 -*-
"""
    The components generator based on the parameter
    decoded from optimizer.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""


from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.networks import *

from brian2 import *


class Generator():
    def __init__(self, parameter):
        self.parameter = parameter

    def generate_block(self, N, ratio):
        block = Block(N, ratio)

    def generate_synapse(self):
        pass

    def generate_reservoir(self):
        pass

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

        self.pathway = Pathway(self.blocks, connect_matrix, model, on_pre, name = 'reservoir_', **kwargs)


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