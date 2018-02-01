import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ------define function------------
def optimal(A, b):
    B = A.T.dot(b)
    AA = np.linalg.inv(A.T.dot(A))
    P = AA.dot(B)
    return P

def lms_test(Data, p):
    l = len(p)
    f = p[l - 1]
    for i in range(len(Data)):
        f += p[i] * Data[i]
    return f

def readout(M, Y):
    one = np.ones((M.shape[0], 1))
    X = np.hstack((M.T, one))
    para = optimal(X, Y)
    return X, para
