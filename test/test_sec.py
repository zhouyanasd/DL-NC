import numpy as np

import src
import src.neuron.neuron as neuron

Spi_neu = neuron.SpikingNeuron(1,'izhikevich_spiking','rate_window')
Spi_neu2 = neuron.SpikingNeuron(2,'izhikevich_spiking','rate_window')


Spi_neu.coding(([1,2]))
Spi_neu2.coding(([2,9]))