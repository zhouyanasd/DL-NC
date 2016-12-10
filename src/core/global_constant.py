import numpy as np

CONNECTION_ARRAY_DTYPE = np.dtype({"names":["connection","weights"],"formats":["i4","f8"]})

INPUT_TIME_WINDOW = 5

TIME_SCALE = 0.01

IZK_INTER_SCALE = 20

INPUT_CONN_RATE = 0.2                      # input connection rate for each input

INTER_RESERVOIR_CONN_RATE = 0.2            # neuron connection rate for each neuron in reservoir

MAX_SYNAPSE_DELAY = 10                     # the max synapse delay for the simulation

