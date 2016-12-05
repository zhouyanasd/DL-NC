import numpy as np

import src
import src.neuron.neuron as neuron
import src.synapse.synapse as synapse

Spi_neu = neuron.SpikingNeuron(1,'izhikevich_spiking','rate_window')
Spi_neu2 = neuron.SpikingNeuron(2,'izhikevich_spiking','rate_window')

# Spi_neu.coding(([1,2]))
# Spi_neu2.coding(([2,9]))

syn = synapse.Synapse(1,Spi_neu,Spi_neu2,0.5)
syn.register()