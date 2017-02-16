import numpy as np

class Sto_state():

    def save_state(self,state):
        np.save('database/state.npy',state)

    def load_sate(self):
        return np.load('database/state.npy')