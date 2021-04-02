import numpy as np
import math

from brian2 import NeuronGroup, Synapses



class BaseFunctions():
    """
    Some basic functions for the simulations

    """

    def data_batch(self, data, n_batch):
        batches = []
        n_data = math.ceil(data.shape[0] / n_batch)
        for i in range(n_batch):
            sub_data = data[i * n_data:(i + 1) * n_data]
            batches.append(sub_data)
        return batches

    def sub_list(self, l, s):
        return [l[x] for x in s]

    def get_sub_dict(self, _dict, *keys):
        return {key:value for key,value in _dict.items() if key in keys}

    def change_dict_key(self, _dict, key1, key2):
        _dict[key2] = _dict.pop(key1)

    def initialize_parameters(self, object, parameter_name, parameter_value):
        '''
         Set the initial parameters of the objects in the block.

         Parameters
         ----------
         object: Brian2.NeuronGroup or Brian2.Synapse, one of the two kinds of objects.
         parameter_name: str, the name of the parameter.
         parameter_value: np.array, the value of the parameter.
         '''
        if '_need_random' in parameter_name:
            parameter_name_ = parameter_name.replace('_need_random','')
            parameter_value_ = np.random.rand(
                    object.pre.variables[parameter_name_].get_value().shape[0]) * parameter_value
        else:
            parameter_name_ = parameter_name
            parameter_value_ = parameter_value
        if isinstance(object, NeuronGroup):
            object.variables[parameter_name_].set_value(parameter_value_)
        elif isinstance(object, Synapses):
            object.pre.variables[parameter_name_].set_value(parameter_value_)

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