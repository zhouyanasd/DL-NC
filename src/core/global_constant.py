import numpy as np

CONNECTION_ARRAY_DTYPE = np.dtype({"names":["connection","weights"],"formats":["i4","f8"]})

INPUT_TIME_WINDOW = 1

READOUT_TIME_WINDOW = 10

OUTPUT_TIME_WINDOW = 30

TIME_SCALE = 0.1

IZK_INTER_SCALE = 100

INPUT_CONN_RATE = 2

INTER_RESERVOIR_CONN_RATE = 10              # neuron connection rate for each neuron in reservoir

MAX_SYNAPSE_DELAY = 10                      # the maximum synapse delay for the simulation

MAX_OPERATION_TIME = 1000                   # the maximum operation time

MAX_WEIGHT = 2.0                            # the max weight for synapse

MIN_WEIGHT = -2.0                           # the min weight for synapse

