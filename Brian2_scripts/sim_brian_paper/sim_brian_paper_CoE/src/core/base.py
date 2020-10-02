import numpy as np
import scipy as sp

from brian2 import NeuronGroup, Synapses

from operator import itemgetter


class MathFunctions():
    """
    Some math functions for the simulations
    """

    def __init__(self):
        pass

    def gamma(self, a, size):
        return sp.stats.gamma.rvs(a, size=size)


class BaseFunctions():
    """
    Some basic functions for the simulations

    """

    def sub_list(self, l, s):
        return [l[x] for x in s]

    def get_sub_dict(self, _dict, *keys):
        values = itemgetter(*keys)(_dict)
        _sub_dict = dict(zip(keys, values))
        return _sub_dict

    def initialize_parameters(self, object, parameter_name, parameter_value):
        '''
         Set the initial parameters of the objects in the block.

         Parameters
         ----------
         object: Brian2.NeuronGroup or Brian2.Synapse, one of the two kinds of objects.
         parameter_name: str, the name of the parameter.
         parameter_value: np.array, the value of the parameter.
         '''

        if isinstance(object, NeuronGroup):
            object.variables[parameter_name].set_value(parameter_value)
        elif isinstance(object, Synapses):
            object.pre.variables[parameter_name].set_value(parameter_value)
        else:
            print('wrong object type')

    def get_parameters_synapse(self, connection_matrix, parameter):
        parameter_list = []
        for index_i, index_j in zip(connection_matrix[0], connection_matrix[1]):
            parameter_list.append(parameter[index_i][index_j])
        return parameter_list

    def connection_matrix_to_adjacent_matrix(self, n, connection_matrix):
        adjacent_matrix = np.zeros(shape=(n, n), dtype='int')
        for a,b in zip(connection_matrix[0],connection_matrix[1]):
            adjacent_matrix[a][b] = 1
        return adjacent_matrix

    def adjacent_matrix_to_connection_matrix(self, adjacent_matrix):
        pass

    def bin2dec(self, binary):
        result = 0
        for i in range(len(binary)):
            result += int(binary[-(i + 1)]) * pow(2, i)
        return result

    def gray2bin(self, gray):
        result = []
        result.append(gray[0])
        for i, g in enumerate(gray[1:]):
            result.append(g ^ result[i])
        return result

    def dec2bin(self,num, l):
        result = []
        while True:
            num, remainder = divmod(num, 2)
            result.append(int(remainder))
            if num == 0:
                break
        if len(result) < l:
            result.extend([0] * (l - len(result)))
        return result[::-1]

    def bin2gary(self, binary):
        result = []
        result.append(binary[0])
        for i, b in enumerate(binary[1:]):
            result.append(b ^ binary[i])
        return result

    def np_extend(self, a, b, axis=0):
        if a is None:
            shape = list(b.shape)
            shape[axis] = 0
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b, axis)

    def np_append(self, a, b):
        shape = list(b.shape)
        shape.insert(0, -1)
        if a is None:
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b.reshape(tuple(shape)), axis=0)

    def vis_block(self, edges):
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        G.add_edges_from(edges)
        values = [node * 0.1 for node in G.nodes()]
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                               node_color=values, node_size=500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True)
        plt.show()