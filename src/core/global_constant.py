import numpy as np

CONNECTION_ARRAY_DTYPE = np.dtype({"names":["connection","weights"],"formats":["i4","f8"]})

INPUT_TIME_WINDOW = 1

TIME_SCALE = 0.1

IZK_INTER_SCALE = 40

INPUT_CONN_RATE = 2                         # input connection rate for each input

INTER_RESERVOIR_CONN_RATE = 10               # neuron connection rate for each neuron in reservoir

MAX_SYNAPSE_DELAY = 10                      # the maximum synapse delay for the simulation

MAX_OPERATION_TIME = 2000                    # the maximum operation time

IN_NEURON_RATE = 0.2                        # the INHIBITORY neuron rate

