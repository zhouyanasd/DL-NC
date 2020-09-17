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
    connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse.
    """

    def __init__(self, N, connect_matrix):
        super().__init__()
        self.N = N
        self.connect_matrix = connect_matrix

    def separate_ex_inh(self, ratio = 4, random_state = None):
        '''
         Generate neural property to separate the ex/inh neurons.

         Parameters
         ----------
         ratio: default int(4), the ratio of excitatory neurons/inhibitory neurons
         random_state: int, the random seed needs to be set when np.shuffle the property.
         '''
        self.ex_inh_ratio = ratio
        self.ex_neurons = int(self.N * (1 / (self.ex_inh_ratio + 1)))
        self.inh_neurons = self.N - self.ex_neurons
        self.neuron_property = np.array(([-1] * self.ex_neurons) + ([1] * self.inh_neurons))
        if random_state != None:
            np.random.seed(random_state)
        np.random.shuffle(self.neuron_property)

    def determine_input_output(self):
        '''
         Determine the index of input and output neurons.
         The input and output are list, e.g. [1,2], [3,4].
         '''

        adjacent_matrix = self.connection_matrix_to_adjacent_matrix(self.N, self.connect_matrix)
        topological_sorting_tarjan = Topological_sorting_tarjan(adjacent_matrix)
        topological_sorting_tarjan.dfs()
        self.input, self.output = topological_sorting_tarjan.suggest_inout()
        self.initialize_parameters(self.neurons, 'property', self.neuron_property)

    def create_neurons(self, model, threshold, reset,refractory, name, **kwargs):
        '''
         Create neurons group for the block.

         Parameters
         ----------
         The parameters follow the necessary 'NeuronGroup' class of Brain2.
         '''

        self.neurons = NeuronGroup(self.N, model, threshold=threshold, reset=reset, refractory=refractory,
                                   method='euler', name = name, **kwargs)
        # try:
        #     self.separate_ex_inh(random_state = kwargs['random_state'])
        # except KeyError:
        #     self.separate_ex_inh()

    def create_synapse(self, model, on_pre, name, **kwargs):
        '''
         Create synapse between neurons for the block.

         Parameters
         ----------
         The parameters follow the necessary 'Synapses' class of Brain2.
         '''

        self.synapses = Synapses(self.neurons, self.neurons, model, on_pre = on_pre,
                                method='euler', name = name, **kwargs)

    def connect(self):
        '''
         Connect neurons using synapse based on the fixed connection matrix.

         Parameters
         ----------
        connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse.
                        The first list is the pre-synapse neurons and the second list is the post-synapse neurons.
         '''

        self.synapse.connect(i = self.connect_matrix[0], j = self.connect_matrix[1])
        self.determine_input_output()

    def initialize(self, NorS, **kwargs):
        '''
         Initialize the parameters of the neurons or synapses.

         Parameters
         ----------
         NorS: self.synapse or self.neurons.
         kwargs: dict{key:str(name), value:np.array[n*1] or np.array[n*n]},
         extensible dict of parameters.
         '''

        for key, value in zip(kwargs.keys(), kwargs.values()):
            self.initialize_parameters(NorS, key, value)

    def join_network(self, net):
        '''
         Let the objects of block join the whole neural network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         '''

        net.add(self.neurons, self.synapse)


class BlockGroup(BaseFunctions):
    """
    This class contains the block set and provides the functions
    for the network with many blocks. This group can be the basic
    components of Reservoir, Encoding and Readout.

    Parameters
    ----------
    N: int, the number of neurons.
    blocks: list[Block], the contained block list.
    """

    def __init__(self):
        super().__init__()
        self.N = 0
        self.blocks = []
        self.blocks_type = []

    def add_block(self, block, type):
        '''
         Add block to the block group.

         Parameters
         ----------
         block: Block, the object of Block.
         '''

        self.blocks.append(block)
        self.blocks_type.append(type)
        self.N += 1

    def initialize(self, parameter_neuron = None, parameter_synapse  = None):
        '''
         Initial the parameters of blocks.

         Parameters
         ----------
         parameter_neuron, parameter_synapse: list[dict], the parameter of
         neuron and synapse.
         '''

        for block, block_type in zip(self.blocks, self.blocks_type):
            if parameter_neuron != None:
                block.initialize(block.neurons, **parameter_neuron[block_type])
            if parameter_synapse != None:
                block.initialize(block.synapses, **parameter_synapse[block_type])

    def join_network(self, net):
        '''
         Let the blocks of the block group join the whole neural network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         '''

        for block in self.blocks:
            block.join_network(net)

class Pathway(BaseFunctions):
    '''
     Pathway contains the synapses between blocks, which
     offer the basic synaptic-like function of connection.

     Parameters
     ----------
     blocks_pre: list[Block], the blocks before pathway.
     blocks_post: list[Block], the blocks after pathway.
     connect_matrix:
     '''

    def __init__(self, blocks_pre, blocks_post, connect_matrix):
        super().__init__()
        self.pre = connect_matrix[0]
        self.post = connect_matrix[1]
        self.blocks_pre = blocks_pre
        self.blocks_post = blocks_post
        self.synapses_group = []

    def create_synapse(self, model, on_pre, on_post,  name = name, **kwargs):
        for index, (index_i, index_j) in enumerate(zip(self.pre, self.post)):
            synapses = Synapses(self.blocks_pre[index_i].neurons, self.blocks_post[index_j].neurons, model, on_pre = on_pre,
                               on_post = on_post, method = 'euler', name = name + str(index), **kwargs)
            self.synapses_group.append(synapses)

    def connect(self):
        for index, synapses in enumerate(self.synapses_group):
            block_pre = self.blocks_pre[self.pre[index]]
            block_post = self.blocks_post[self.post[index]]
            connect_matrix = self.np_two_combination(block_pre.output, block_post.input)
            synapses.connect(i = connect_matrix[0], j = connect_matrix[1])

    def _initialize(self, synapses, **kwargs):
        for key, value in zip(kwargs.keys(), kwargs.values()):
            converted_value = self.get_parameters(self.connect_matrix, value)
            self.initialize_parameters(synapses, key, converted_value)

    def initialize(self, parameter_synapse):
        for synapses in self.synapse:
            self._initialize(synapses, **parameter_synapse)

    def join_network(self, net):
        '''
         Let the synapse of pathway join the whole neural network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         '''

        for synapse in self.synapse:
            net.add(synapse)


class Reservoir(BaseFunctions):
    """
    This class offers a basic property and functions of reservoir containing blocks.

    Parameters
    ----------
    block_group: BlockGroup, the block group in this reservoir.
    pathway: Pathway, the pathway between blocks.
    """

    def __init__(self):
        super().__init__()
        self.block_group = None
        self.pathway = None

    def register_input_output(self, o, i):
        '''
         Determine the index of input and output neurons.

         Parameters
         ----------
         '''

        self.input = o
        self.output = i

    def register_blocks(self, block_group):
        '''
         Register block group to the reservoir.

         Parameters
         ----------
         block_group: BlockGroup, the object of BlockGroup.
         '''

        self.block_group = block_group

    def register_pathway(self, pathway):
        '''
         Register pathway to the reservoir.

         Parameters
         ----------
         pathway: Pathway, the object of Pathway.
         '''

        self.pathway = pathway

    def connect(self):
        '''
         Connect blocks base on the synapses in the reservoir.

         Parameters
         ----------
         '''

        self.pathway.connect()

    def initialize(self, parameter_block_neurons, parameter_block_synapses, parameter_pathway):
        self.block_group.initialize(parameter_block_neurons,parameter_block_synapses)
        self.pathway.initialize(parameter_pathway)

    def join_network(self, net):
        '''
         Let the objects of reservoir join the whole neural network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         '''

        self.block_group.join_network(net)
        self.pathway.join_network(net)


class LSM_Network(BaseFunctions):
    """
    This class offers a basic property and functions of network
    containing reservoir, encoding and readout.

    Parameters
    ----------
    layers: dict, contained layers.
    pathways: list[Pathway], the list of Pathway between layers.
    """

    def __init__(self):
        super().__init__()
        self.layers = {}
        self.pathways = {}

    def register_layer(self, layer, name):
        '''
         Register layers in the network.

         Parameters
         ----------
         layer: Reservoir or BlockGroup, the layer in network.
         name: str, the name for the dict.
         '''

        self.layers[name] = layer
        try:
            self.add(layer)
        except TypeError:
            layer.join_network(self)

    def register_pathway(self, pathway, name):
        '''
         Add pathway to the reservoir.

         Parameters
         ----------
         pathway: Pathway, the object of Pathway.
         name: str, the name for the dict.
         '''

        self.pathway[name] = pathway

    def connect(self):
        '''
         Connect layers with the pathway between them.

         Parameters
         ----------
         '''
        for pathway in self.pathways:
            pathway.connect()

    def initialize(self, **parameter):
        '''
         Initialize all the parameters in the network.

         Parameters
         ----------
         **parameter: dict{key:str, value:list}
         '''
        for key, layer in zip(self.layers.keys(), self.layers.values()):
            layer.initialize(parameter[key])
        for key, pathway in zip(self.pathway.keys(), self.pathway.values()):
            pathway.initialize(parameter[key])

    def join_network(self, net):
        '''
         Let the layer and pathway join the whole neural network.

         Parameters
         ----------
         '''
        for layer in self.layers.values():
            layer.join_network(net)
        for pathway in self.pathways:
            pathway.join_network(net)
