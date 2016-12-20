import src
import numpy as np
import matplotlib.pyplot as plt

class Visualization(object):

    def __init__(self, total_time):
        self.t = np.arange(0,total_time)

    def show(self):
        plt.show()

    def I(self,neuron):
        I = neuron.I
        print(I)
        plt.figure()
        plt.plot(self.t[:],I)
        plt.show()


    def add_fired_fig(self,fig,Liquid):
        ax = fig.add_subplot(3, 1, 2)
        for j in range(Liquid.reservoir_list[0].neuron_list.size):
            fired = Liquid.reservoir_list[0].neuron_list[j].fired_sequence
            for i in fired:
                ax.scatter(i,0.5*j,alpha=.5)



