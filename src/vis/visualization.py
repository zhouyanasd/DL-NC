import src
import numpy as np
import matplotlib.pyplot as plt

class Visualization(object):

    def __init__(self, total_time):
        self.t = np.arange(0,total_time)

    def show_I(self,neuron):
        I = neuron.I
        print(I)
        plt.figure()
        plt.plot(self.t[:],I)
        plt.show()

