from brian2 import *
import copy as c
import os
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)

net = Network()
a = 2
b = c.deepcopy(net)

print(os.getpid(),id(b), id(net))