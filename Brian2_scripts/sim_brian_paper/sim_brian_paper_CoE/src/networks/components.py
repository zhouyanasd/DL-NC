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
                    The first list is the pre-synapse neurons and the second list is the post-synapse neurons.
    """

    def __init__(self, N, connect_matrix):
        super().__init__()
        self.N = N
        self.connect_matrix = connect_matrix.astype(int)

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
        self.inh_neurons = int(self.N - self.ex_neurons)
        self.neuron_property = np.array(([-1] * self.ex_neurons) + ([1] * self.inh_neurons))
        if random_state != None:
            np.random.seed(random_state)
        np.random.shuffle(self.neuron_property)
        self.initialize_parameters(self.neurons, 'property', self.neuron_property)

    def determine_input_output(self):
        '''
         Determine the index of input and output neurons.
         The input and output are list, e.g. [1,2], [3,4].
         '''

        adjacent_matrix = self.connection_matrix_to_adjacent_matrix(self.N, self.connect_matrix)
        topological_sorting_tarjan = Topological_sorting_tarjan(adjacent_matrix)
        topological_sorting_tarjan.dfs()
        self.input, self.output = topological_sorting_tarjan.suggest_inout()

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

    def create_synapse(self, model, on_pre, on_post, name, **kwargs):
        '''
         Create synapse between neurons for the block.

         Parameters
         ----------
         The parameters follow the necessary 'Synapses' class of Brain2.
         '''

        self.synapses = Synapses(self.neurons, self.neurons, model, on_pre = on_pre, on_post = on_post,
                                method='euler', name = name, **kwargs)

    def connect(self):
        '''
         Connect neurons using synapse based on the fixed connection matrix.

         Parameters
         ----------
        connect_matrix: list[list[int], list[int]], the fixed connection matrix for inner synapse.
         '''

        self.synapses.connect(i = self.connect_matrix[0], j = self.connect_matrix[1])

    def initialize(self, component, **kwargs):
        '''
         Initialize the parameters of the neurons or synapses.

         Parameters
         ----------
         component: self.synapse or self.neurons.
         kwargs: dict{key:str(name), value:np.array[n*1] or np.array[n*n]},
         extensible dict of parameters.
         '''

        for key, value in zip(kwargs.keys(), kwargs.values()):
            self.initialize_parameters(component, key, value)

    def join_network(self, net):
        '''
         Let the objects of block join the whole neural network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         '''

        net.add(self.neurons, self.synapses)


class BlockGroup(BaseFunctions):
    """
    This class contains the block set and provides the functions
    for the network with many blocks. This group can be the basic
    components of Reservoir, Encoding and Readout.

    Parameters
    ----------
    N: int, the number of neurons.
    blocks: list[Block], the contained block list.
    blocks_type: list[int], the block type order according to 'structure_blocks'.
    """

    def __init__(self):
        super().__init__()
        self.N = 0
        self.blocks = []
        self.blocks_type = []

    def get_neurons_count(self, blocks = None):
        '''
         Count the total number of neurons int the blocks.

         Parameters
         ----------
         blocks: list[Block], the block list for counting.
         '''

        if blocks == None:
            blocks_ = self.blocks
        else:
            blocks_= blocks
        N_neurons = 0
        for block in blocks_:
            N_neurons += self.blocks[block].N
        return N_neurons

    def add_block(self, block, type):
        '''
         Add block to the block group.

         Parameters
         ----------
         block: Block, the object of Block.
         type: int, the block type order according to 'structure_blocks'.
               -1 represents the encoding and readout.
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
     connect_matrix: list[list[int], list[int]], the fixed connection matrix between 'Block_group',
                     int for 'Block'.
     synapses_group: list[Brian.Synapse], the built 'Synapse'.
     connect_type: str, the connection type for the synapses of the Pathway.
     '''

    def __init__(self, blocks_pre, blocks_post, connect_matrix):
        super().__init__()
        self.connect_matrix = connect_matrix.astype(int)
        self.pre = connect_matrix[0]
        self.post = connect_matrix[1]
        self.blocks_pre = blocks_pre
        self.blocks_post = blocks_post
        self.synapses_group = []
        self.connect_type = 'full'

    def _set_connect_type(self, type):
        '''
         Set the connection type of this pathway.

         Parameters
         ----------
         type: str, the connection type for the synapses of the Pathway.
         '''

        if type in ['full', 'one_to_one']:
            self.connect_type = type
        else:
            raise ("wrong connection type, only 'full' or 'one_to_one'.")

    def create_synapse(self, model, on_pre, on_post,  name, **kwargs):
        '''
         Create synapse between neurons for the Pathway.

         Parameters
         ----------
         The parameters follow the necessary 'Synapses' class of Brain2.
         '''

        for index, (index_i, index_j) in enumerate(zip(self.pre, self.post)):
            synapses = Synapses(self.blocks_pre[index_i].neurons, self.blocks_post[index_j].neurons, model, on_pre = on_pre,
                               on_post = on_post, method = 'euler', name = name + str(index), **kwargs)
            self.synapses_group.append(synapses)

    def connect(self):
        '''
         Connect neurons using synapse based on the fixed connection matrix and connection type.

         Parameters
         ----------
         The parameters follow the necessary 'Synapses' class of Brain2.
         '''

        if self.connect_type == 'full':
            for index, synapses in enumerate(self.synapses_group):
                block_pre = self.blocks_pre[self.pre[index]]
                block_post = self.blocks_post[self.post[index]]
                connect_matrix = [[],[]]
                for i in block_pre.output:
                    for j in block_post.input:
                        connect_matrix[0].append(i)
                        connect_matrix[1].append(j)
                synapses.connect(i = connect_matrix[0], j = connect_matrix[1])
        elif self.connect_type == 'one_to_one':
            count = 0
            for index, synapses in enumerate(self.synapses_group):
                block_pre = self.blocks_pre[self.pre[index]]
                block_post = self.blocks_post[self.post[index]]
                synapses.connect(i = block_pre.output, j = block_post.input[count:count+len(block_pre.output)])
                count = count + len(block_pre.output)


    def _initialize(self, synapses, **kwargs):
        '''
         Initialize the signal synapse.

         Parameters
         ----------
         synapses: Brian2.Synapse, The 'Synapses' class of Brain2.
         kwargs: dict, the parameters.
         '''

        for key, value in zip(kwargs.keys(), kwargs.values()):
            self.initialize_parameters(synapses, key, value)

    def initialize(self, **parameter_synapse):
        '''
         Initialize the synapses of the Pathway.

         Parameters
         ----------
         parameter_synapse: dict, the parameters.
         '''

        for synapses in self.synapses_group:
            self._initialize(synapses, **parameter_synapse)

    def join_network(self, net):
        '''
         Let the synapse of pathway join the whole neural network.

         Parameters
         ----------
         net: Brian2.Network, the existing neural network.
         '''

        for synapse in self.synapses_group:
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

    @property
    def input_neurons_count(self):
        '''
         Get the total number of neurons of the input.

         Parameters
         ----------
         '''
        return self.block_group.get_neurons_count(self.input)

    @property
    def output_neurons_count(self):
        '''
         Get the total number of neurons of the output.

         Parameters
         ----------
         '''
        return self.block_group.get_neurons_count(self.output)

    def register_input_output(self, o, i):
        '''
         Determine the index of input and output neurons.

         Parameters
         ----------
         '''

        self.input = i
        self.output = o

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

    def initialize(self, parameter_block_neurons, parameter_block_synapses, parameter_pathway):
        '''
        Initialize the block group and pathway in the reservoir.

         Parameters
         ----------
         parameter_block_neurons: dict{dict{dict}}, the parameters for neurons of blocks.
         parameter_block_synapses: dict{dict{dict}}, the parameters for synapses of blocks.
         parameter_pathway: dict, the parameters for pathway.
         '''
        self.block_group.initialize(parameter_block_neurons,parameter_block_synapses)
        self.pathway.initialize(**parameter_pathway)

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

    def register_pathway(self, pathway, name):
        '''
         Add pathway to the reservoir.

         Parameters
         ----------
         pathway: Pathway, the object of Pathway.
         name: str, the name for the dict.
         '''

        self.pathways[name] = pathway

    def _initialize(self, components, **parameter):
        '''
         Initialize all the parameters in the network.

         Parameters
         ----------
         **parameter: dict{key:str, value:list}
         '''
        for key, component in zip(components.keys(), components.values()):
            if parameter[key] != None:
                component.initialize(**parameter[key])

    def initialize(self, **parameter):
        '''
         Initialize all the parameters in the network.

         Parameters
         ----------
         **parameter: dict{key:str, value:list}

         Examples
         ----------
         {'encoding': None,
          'encoding_reservoir': {'plasticity': 0.7, 'strength': 0.7', 'type: 1.0'},
          'readout': None,
          'reservoir_readout': None,
          'reservoir': {'parameter_block_neurons': {'hierarchy': {'tau': 0.6, 'threshold': 0.6},
                                                    'random': {'tau': 0.3, 'threshold': 0.3}},
                        'parameter_block_synapses': {'hierarchy': {'plasticity': 0.6, 'strength': 0.6, 'type': 1.0},
                                                     'random': {'plasticity': 0.3, 'strength': 0.3, 'type': 1.0}},
                        'parameter_pathway': {'type: 1.0', 'plasticity': 0.2, 'strength': 0.2}}}
         '''
        self._initialize(self.layers, **parameter)
        self._initialize(self.pathways, **parameter)

    def join_network(self, net):
        '''
         Let the layer and pathway join the whole neural network.

         Parameters
         ----------
         '''
        for layer in self.layers.values():
            layer.join_network(net)
        for pathway in self.pathways.values():
            pathway.join_network(net)
