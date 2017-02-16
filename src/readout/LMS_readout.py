import numpy as np
from scipy.optimize import leastsq

from src.readout import Readout

class LMS_readout(Readout):
    def __int__(self, id):
        Readout.__init__(self, id)

    def LMS_train(self, label):
        p = np.random.randn(self.read_number, self.pre_state.shape[0])
        Para = 0
        for i in self.read_number:
            Para = leastsq(self.__error, p[i], args=(label[i], self.pre_state))
        return Para[0]

    def LMS_test(self):
        pass

    def __error(self,p,y, args):
        f = 0

        for i in range (len(args)):
            f += p[i]*args[i]
        return f-y

