# -*- coding: utf-8 -*-
"""
    The fundamental neurons and network structure
    including local blocks.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""


from brian2 import *

class Block():
    """Some basic function for data transformation or calculation.

    This class offers ....

    Parameters
    ----------
    N: the number of neurons
    Group
    Synapse

    Functions
    ----------

    """
    def __init__(self, N):
        pass


class Neuron():
    """Some basic function for data transformation or calculation.

    This class offers ....

    Parameters
    ----------
    property: 'ex' or 'inh'

    """
    def __init__(self, property):
        self.property = property