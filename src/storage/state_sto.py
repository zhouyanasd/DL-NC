import numpy as np

class Sto_state():

    def save_state(self,state, name):
        np.save('database/readout_state'+str(name)+'.npy',state)

    def load_sate(self,name):
        return np.load('database/readout_state'+str(name)+'.npy')