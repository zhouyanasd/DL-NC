import numpy as np

from..core import Base,TIME_SCALE

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

    def Tri_function(self):
        rng = np.random
        def chose_fun():
            fun_type = np.arange(3)
            rng.shuffle(fun_type)
            fun = fun_type[:1]
            return fun
        def change_fun(rate):
            fun_type = np.arange(100)
            rng.shuffle(fun_type)
            fun = fun_type[:1]
            if fun > 100*rate:
                return False
            else:
                return True
        t = np.arange(self.in_number)
        data = []
        for i in (self.group):
            for j in t:
                if change_fun(0.05):


                if chose_fun() == 0:
                    sin = np.sin(t*TIME_SCALE)
                    data_t = sin
        data.append(data_t)
        return data

