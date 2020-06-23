# -*- coding: utf-8 -*-
"""
    The fundamental algorithms used in CoE.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

import numpy as np
from enum import Enum
from collections import OrderedDict


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
        _g = self.g
        for c_root in self._sorting: # 这里先将链接矩阵中所有强连通分量的链接删除
            c_sub = self.components[c_root]
            X, Y = np.meshgrid(c_sub, c_sub)
            for x, y in zip(X.reshape(-1, ).tolist(), Y.reshape(-1, ).tolist()):
                _g[x][y] = 0
            _g_input = np.sum(self.g, 0)[c_sub] # 然后统计各个强连通分量的出入链接数
            _g_output = np.sum(self.g, 1)[c_sub]
            if sum(_g_input) == 0: # 将入/出度为0的分量（包括只有一个节点的）作为输入/出（多节点随机选一个）
                _c_sub = c_sub.copy()
                np.random.shuffle(_c_sub)
                incoming.append(_c_sub[0])
            if sum(_g_output) == 0:
                _c_sub = c_sub.copy()
                np.random.shuffle(_c_sub)
                outgoing.append(_c_sub[0])
        return incoming, outgoing