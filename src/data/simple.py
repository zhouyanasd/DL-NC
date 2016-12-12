import numpy as np

from..core import Base

class Simple(Base):
    def __init__(self, in_feature, in_number, group):
        self.in_feature = in_feature
        self.in_number = in_number
        self.group = group

    def Possion(self):
        rng = np.random
        data = []
        for i in range (self.group):
            data_t = rng.poisson(i+1,(self.in_feature,self.in_number))
            data.append(data_t)
        return data


