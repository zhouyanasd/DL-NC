from .import Synapse

import numpy as np

class izk_synapse(Synapse):
    def __init__(self, id, pre_neuron, post_neuron, delay, weight,*args, **kwargs):
        Synapse.__init__( id, pre_neuron, post_neuron, delay, weight)
        self.A_plus = 0.012
        self.A_minus = -0.012
        self.T_plus = 20.0
        self.T_minus = 40.0

    def adjust_weight(self):
        if (self.spiking_buffer[0] == 1):
            self._last_arrive_time = self.get_global_time()
            if (self._last_spiking_time == 0): return self.weight
            TDelay = self._last_spiking_time - self._last_arrive_time
            if (TDelay > 0): return self.weight
            self.d_w += self.A_minus * np.exp(-np.abs(TDelay)/self.T_minus)

        if (self.post_neuron.fired):
            TDelay = self.get_global_time() - self._last_arrive_time
            if (TDelay < 0): return self.weight
            self.d_w += self.A_plus * np.exp( TDelay / self.T_minus)

        self.update()





