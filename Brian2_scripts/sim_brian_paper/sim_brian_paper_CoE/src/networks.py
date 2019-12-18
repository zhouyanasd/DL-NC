# -*- coding: utf-8 -*-
"""
    The fundamental neurons and network structure
    including local blocks.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""


from brian2 import *

class Block():
    """
    This class offers a basic property and functions of block in LSM.

    Parameters
    ----------
    N: int, the number of neurons
    connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse.
                    The first list is the pre-synapse neurons and the second list is the post-synapse neurons.
    """

    def __init__(self, N):
        self.N = N
        self.connect_matrix = None

    def create_neurons(self, model, threshold, reset,refractory, name, **kwargs):
        '''
         Create neurons group for the block.

         Parameters
         ----------
         The parameters follow the necessary 'NeuronGroup' class of Brain2.
         '''
        self.neurons = NeuronGroup(self.N, model, threshold=threshold, reset=reset, refractory=refractory,
                                   method='euler', name = name, **kwargs)

    def create_synapse(self, model, on_pre, delay, name, **kwargs):
        '''
         Create synapse between neurons for the block.

         Parameters
         ----------
         The parameters follow the necessary 'Synapses' class of Brain2.
         '''
        self.synapse = Synapses(self.neurons, self.neurons, model, on_pre = on_pre, delay = delay,
                                method='euler', name = name, **kwargs)

    def connect(self, connect_matrix):
        '''
         Connect neurons using synapse based on the fixed connection matrix.

         Parameters
         ----------
         connect_matrix: list[list[int], list[int]], the fixed connection matrix.
         '''
        self.connect_matrix = connect_matrix
        self.synapse.connect(i = connect_matrix[0], j = connect_matrix[1])

    def join_networks(self, net):
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

         Parameters
         ----------
         '''
        self.input = [1, 2]
        self.output = [8, 9]


class Reservoir():
    """

    This class offers a basic property and functions of reservoir containing blocks.

    Parameters
    ----------
    N: int, the number of blocks
    connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse between blocks.
                    The first list is the pre-synapse blocks and the second list is the post-synapse blocks.
    blocks: list[Block], the list of block
    synapses: list[Brain.Synapse], the list of synapse between blocks
    """

    def __init__(self, N):
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

    def create_synapse(self, block_pre, block_post, model, on_pre, delay, name, **kwargs):
        '''
         Create synapses between blocks.

         Parameters
         ----------
         block_pre: Block, the block before the synapse
         block_post Block, the block after the synapse
         Other parameters follow the necessary 'Synapses' class of Brain2.
         '''
        synapse = Synapses(block_pre.neurons, block_post.neurons, model, on_pre=on_pre, delay=delay,
                                method='euler', name = name, **kwargs)
        self.synapses.append(synapse)

    def connect_blocks(self, synapse):
        for block in self.blocks:
            block.determine_input_output()
        synapse.connect(i = [], j = [])

    def join_network(self, net):
            for block in self.blocks:
                block.join_networks(net)
            net.add(*self.synapses)

    # def create_blocks(self, N_blocks, neurons_block, dynamics_neurons, dynamics_synapse,
    #                   dynamics_synapse_pre, threshold, reset, refractory, delay, connect_matrix,
    #                   name_neurons, name_synapses):
    #
    #     for index in range(N_blocks):
    #         block = Block(neurons_block)
    #         block.create_neurons(dynamics_neurons, threshold, reset, refractory,
    #                                        name_neurons)
    #         block.create_synapse(dynamics_synapse, dynamics_synapse_pre, delay,
    #                                        name_synapses)
    #         block.connect(connect_matrix)
    #         self.add_block(block)