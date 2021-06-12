# -*- coding: utf-8 -*-
"""
    The fundamental algorithms used in CoE.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions

from enum import Enum
from collections import OrderedDict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class TraversalState(Enum):
    WHITE = 0
    GRAY = 1
    BLACK = 2


class DFS():
    """
    Depth-first search for a given graph.

    Parameter
    -----
    g: the adjacency matrix of a graph, store as a two-dim n*n numpy.array.
    The first dim means the source and the second dim means the target.
    Example, `g[1][2] = 1` means the connection from 1 to 2 is exist.

    Notes
    -----
    The property of nodes are store in the one-dim numpy.array.
    Example, `discovery[1] = 12` means the discovery time of 1 is 12.


    See also
    --------
    [1]. Introduction to algorithms[M]. MIT press, 2009.
    """
    def __init__(self, g):
        self.time = 0 # 记录搜索次数
        self.g = g
        self.n = len(self.g)
        self.color = np.array([TraversalState.WHITE] * self.n) # 节点状态
        self.pi = np.zeros(self.n) - 1 # 前驱节点
        self.discovery = np.zeros(self.n) # 发现时间
        self.finishing = np.zeros(self.n) # 完成时间
        self.show_actions = False

    def dfs(self):
        N = np.arange(self.n) # 随机找到一个根节点开始搜索
        np.random.shuffle(N)
        for u in N: # 循环对节点进行搜索，当节点未被访问时，进行搜索
            if self.color[u] == TraversalState.WHITE:
                self.visit(u)

    def visit(self, u):
        self.time += 1
        self.discovery[u] = self.time
        self.color[u] = TraversalState.GRAY
        self.show(u, 'gray', self.time)
        for v, con in enumerate(self.g[u]): # 根据链接找后驱节点，当没被访问时，递归调用，进行搜索
            if con == 1:
                if self.color[v] == TraversalState.WHITE:
                    self.pi[v] = u
                    self.visit(v)
        self.color[u] = TraversalState.BLACK # 结束搜索，回溯操作，记录结束时间
        self.show(u, 'black', self.time)
        self.time += 1
        self.finishing[u] = self.time

    def show(self, *args):
        if self.show_actions is True:
            print(*args)
        else:
            pass


class Tarjan(DFS):
    """
    The Tarjan algorithm finding strongly connected components,
    based on Depth-first search for a given graph.

    Parameter
    -----
    g: the adjacency matrix of a graph, store as a two-dim n*n numpy.array.
    The first dim means the source and the second dim means the target.
    Example, `g[1][2] = 1` means the connection from 1 to 2 is exist.

    Notes
    -----
    The property of nodes are store in the one-dim numpy.array.
    Example, `discovery[1] = 12` means the discovery time of 1 is 12.


    See also
    --------
    [1]. Introduction to algorithms[M]. MIT press, 2009.
    [2]. Tarjan, RE, Depth-first search and linear graph algorithms,
         SIAM Journal on Computing, 1972, 1 (2): 146–160, doi:10.1137/0201010
    """
    def __init__(self, g):
        super(Tarjan, self).__init__(g)
        self.low = np.zeros(self.n)
        self.stack = [] # 引入一个搜索栈
        self.components = OrderedDict() # 引入顺序字典

    def visit(self, u):
        self.time += 1
        self.discovery[u] = self.low[u] = self.time
        self.stack.append(u) # 与DFS相比，这里多了一个入栈
        self.color[u] = TraversalState.GRAY
        self.show(u, 'gray', self.time)
        for v, con in enumerate(self.g[u]):
            if con == 1:
                if self.color[v] == TraversalState.WHITE:
                    self.pi[v] = u
                    self.visit(v)
                    self.low[u] = min(self.low[u], self.low[v]) # 回溯时，比较当前点的low值与后驱节点的low值取最小
                elif self.color[v] == TraversalState.GRAY: # 如果是正在访问的节点则比较一个discovery，取小值
                    if (self.discovery[v] < self.low[u]):
                        self.low[u] = self.discovery[v]
        if (self.discovery[u] == self.low[u] and len(self.stack) != 0): # 回溯时出栈
            self.show("********连通图********")
            m = self.stack.pop() # 首先栈顶出栈，并随后记为结束搜索
            self.color[m] = TraversalState.BLACK
            self.time += 1
            self.finishing[m] = self.time
            self.components[u] = [m] # 将改节点作为一个联通图的根节点
            self.show(m, 'black', self.time)
            while m != u and len(self.stack) != 0: # 如果节点u的后续栈中还有元素则继续弹出直到u为止
                m = self.stack.pop()
                self.components[u].append(m)
                self.color[m] = TraversalState.BLACK
                self.time += 1
                self.finishing[m] = self.time
                self.show(m, 'black', self.time)
            self.show("**********************")


class Topological_sorting_tarjan(Tarjan):
    """
        The Tarjan algorithm based Topological sorting algorithm.
        The Topological sorting is the inverted order of the finishing order.
        The strongly connected components are seen as one node.

        """
    def __init__(self, g):
        super(Topological_sorting_tarjan, self).__init__(g)

    @property
    def _sorting(self):
        return np.array(list(self.components.keys())[::-1])

    def topological_sorting(self):
        components = list(self.components.keys())
        finishing = self.finishing[components]
        self.sorting = np.array(components)[finishing.argsort()[::-1]]

    def suggest_inout(self):  # 因为拓扑排序一定有开始和结束，DAG是单方向的，所以直接找到入度为0的点作为输入，出度为0的点作为输出
        incoming = []
        outgoing = []
        _g = self.g.copy()
        for c_root in self._sorting: # 这里先将链接矩阵中所有强连通分量的链接删除
            c_sub = self.components[c_root]
            X, Y = np.meshgrid(c_sub, c_sub)
            for x, y in zip(X.reshape(-1, ).tolist(), Y.reshape(-1, ).tolist()):
                _g[x][y] = 0
            _g_input = np.sum(_g, 0)[c_sub] # 然后统计各个强连通分量的出入链接数
            _g_output = np.sum(_g, 1)[c_sub]
            if sum(_g_input) == 0: # 将入/出度为0的分量（包括只有一个节点的）作为输入/出（多节点随机选一个）
                _c_sub = c_sub.copy()
                np.random.shuffle(_c_sub)
                incoming.append(_c_sub[0])
            if sum(_g_output) == 0:
                _c_sub = c_sub.copy()
                np.random.shuffle(_c_sub)
                outgoing.append(_c_sub[0])
        return incoming, outgoing

    def suggest_inout_multi_io(self, multi_io = 0.2):  # 因为拓扑排序一定有开始和结束，DAG是单方向的，所以直接找到入度为0的点作为输入，出度为0的点作为输出
        incoming = []
        outgoing = []
        _g = self.g.copy()
        for c_root in self._sorting: # 这里先将链接矩阵中所有强连通分量的链接删除
            c_sub = self.components[c_root]
            X, Y = np.meshgrid(c_sub, c_sub)
            for x, y in zip(X.reshape(-1, ).tolist(), Y.reshape(-1, ).tolist()):
                _g[x][y] = 0
            _g_input = np.sum(_g, 0)[c_sub] # 然后统计各个强连通分量的出入链接数
            _g_output = np.sum(_g, 1)[c_sub]
            if sum(_g_input) == 0: # 将入/出度为0的分量（包括只有一个节点的）作为输入/出（多节点随机选一个）
                _c_sub = c_sub.copy()
                np.random.shuffle(_c_sub)
                incoming.append(_c_sub[0])
            if sum(_g_output) == 0:
                _c_sub = c_sub.copy()
                np.random.shuffle(_c_sub)
                outgoing.append(_c_sub[0])
                # 如果总的输入输出节点太少，那么会增加作为输入输出的节点
                # 如果不够那么就随机选取一些
        if len(incoming)<=int(multi_io*self.n) :
            unpicked = np.setdiff1d(np.arange(self.n), incoming)
            np.random.shuffle(unpicked)
            incoming.extend(list(unpicked[:int(multi_io*self.n-len(incoming))]))
        if len(outgoing)<=int(multi_io*self.n) :
            unpicked = np.setdiff1d(np.arange(self.n), outgoing)
            np.random.shuffle(unpicked)
            outgoing.extend(list(unpicked[:int(multi_io*self.n-len(outgoing))]))
        return incoming, outgoing


class Direct_scale_free(BaseFunctions):
    def __init__(self, init_nodes, final_nodes, alpha=0.4, beta=0.2,
                 gama=0.4, delta_in=0, delta_out=0):
        self.init_nodes = init_nodes
        self.final_nodes = final_nodes
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.delta_in = delta_in
        self.delta_out = delta_out
        self.nodes = list(np.arange(self.init_nodes))
        self.o, self.i = self._full_connected(self.init_nodes)
        self.new_node = self.nodes[-1] + 1

    def _full_connected(self, init_nodes):
        nodes = list(np.arange(init_nodes))
        o, i = [], []
        for node_pre in nodes:
            for node_post in nodes:
                if node_pre == node_post:
                    continue
                else:
                    i.append(node_post)
                    o.append(node_pre)
        return o, i

    def _check_edge_exist(self, post_node, pre_note):
        for edge in self.edges:
            if pre_note == edge[0] and post_node == edge[1]:
                return True
            else:
                return False

    def _check_all_edges_exist(self, pre_notes, post_nodes):
        for pre_note in pre_notes:
            for post_node in post_nodes:
                if self._check_edge_exist(pre_note, post_node):
                    continue
                else:
                    return False

    def _get_absence_edges(self):
        pass

    @property
    def edges(self):
        return list(zip(self.o, self.i))

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return len(self.edges)

    def get_degree_in(self, node):
        return sum(np.array(self.i) == node)

    def get_degree_out(self, node):
        return sum(np.array(self.o) == node)

    def rand_prob_node(self):
        node_probe_in = []
        node_probe_out = []
        for node in self.nodes:
            degree_in = self.get_degree_in(node)
            degree_out = self.get_degree_out(node)
            node_probe_in.append((degree_in + self.delta_in) / (self.num_edges + self.delta_in * self.num_nodes))
            node_probe_out.append(
                (degree_out + self.delta_out) / (self.num_edges + self.delta_out * self.num_nodes))
        random_probe_node_in = np.random.choice(self.nodes, p=node_probe_in)
        random_probe_node_out = np.random.choice(self.nodes, p=node_probe_out)
        return random_probe_node_in, random_probe_node_out

    def add_edge(self):
        r = np.random.rand()
        if self.num_nodes <= 1:
            self.add_node()
        if self.num_edges == 0:
            nodes = self.nodes.copy()
            np.random.shuffle(nodes)
            self.i.append(nodes[0])
            self.o.append(nodes[1])
        if r > 0 and r <= self.alpha:
            random_probe_node_in, random_probe_node_out = self.rand_prob_node()
            self.i.append(random_probe_node_in)
            self.o.append(self.new_node)
            self.add_node()
        if r > self.alpha and r <= self.beta + self.alpha:
            random_probe_node_in, random_probe_node_out = self.rand_prob_node()
            if not self._check_edge_exist(random_probe_node_in, random_probe_node_out):
                self.i.append(random_probe_node_in)
                self.o.append(random_probe_node_out)
        if r > self.beta + self.alpha and r <= 1:
            random_probe_node_in, random_probe_node_out = self.rand_prob_node()
            self.i.append(self.new_node)
            self.o.append(random_probe_node_out)
            self.add_node()

    def add_node(self):
        self.nodes.append(self.new_node)
        self.new_node += 1

    def generate_graph(self):
        while self.new_node != self.final_nodes:
            self.add_edge()


class Evaluation():
    """
    Some basic function for evaluate the output of LSM.

    This class offers evaluation functions for learning tasks.
    """

    def __init__(self):
        pass

    def readout_sk(self, X_train, X_validation, X_test, y_train, y_validation, y_test, **kwargs):
        lr = LogisticRegression(**kwargs)
        lr.fit(X_train.T, y_train.T)
        y_train_predictions = lr.predict(X_train.T)
        y_validation_predictions = lr.predict(X_validation.T)
        y_test_predictions = lr.predict(X_test.T)
        return accuracy_score(y_train_predictions, y_train.T), \
               accuracy_score(y_validation_predictions, y_validation.T), \
               accuracy_score(y_test_predictions, y_test.T)