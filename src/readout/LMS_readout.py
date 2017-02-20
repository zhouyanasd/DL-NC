import numpy as np
from scipy.optimize import leastsq

from src.readout import Readout

class LMS_readout(Readout):

    def __init__(self, id):
        Readout.__init__(self, id)
        self.para_list = []

    def LMS_train(self, label):
        p = np.random.randn(self.read_number, self.coded_state.shape[0])
        for i in range(self.read_number):
            para = leastsq(self.__error, p[i], args = (label[i], self.coded_state))
            self.para_list.append(para[0])

    def LMS_test(self):
        Output_list = []
        Output = np.zeros(self.read_number)
        for i in range(self.read_number):
            Output = self.para_list[i].dot(self.coded_state)
            Output_list.append(Output)
        return Output_list

    def __error(self,p,y, args):
        f = 2                      #bias
        for i in range (len(args)):
            f += p[i]*args[i]
        return f-y

