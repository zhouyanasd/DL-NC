import numpy as np

from src.readout import Readout

class LMS_readout(Readout):
    def __int__(self, id):
        Readout.__init__(self, id)
        self.LMS_num = 0

    def add_LMS(self,class_num):
        self.LMS_num += class_num



    def LMS_train(self,all_state,):
        pass

    def LMS_test(self):
        pass


# class a():
#     def aa(self):
#         print('a')
#
# class b():
#     def aa(self):
#         print('b')
#
# B=b()
# B.aa()