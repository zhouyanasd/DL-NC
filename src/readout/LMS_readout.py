import numpy as np
from scipy.optimize import leastsq

from src.readout import Readout

class LMS_readout(Readout):
    def __int__(self, id):
        Readout.__init__(self, id)
        self.Para_list = []

    def LMS_train(self, label):
        p = np.random.randn(self.read_number, self.pre_state.shape[0])
        for i in self.read_number:
            Para = leastsq(self.__error, p[i], args=(label[i], self.pre_state))
            self.Para_list.append(Para[0])

    def LMS_test(self, Data_test):
        Output = np.zeros(self.read_number)
        for i in self.read_number:
            Output[i] = self.Para_list[i].dot(Data_test)
        return Output

    def __error(self,p,y, args):
        f = 1                      #bias
        for i in range (len(args)):
            f += p[i]*args[i]
        return f-y

