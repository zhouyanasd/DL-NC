# -*- coding: utf-8 -*-
"""
    The functions for preparing the data.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

import numpy as np

class UCI_classification():
    def __init__(self):
        pass


    def loadData(self, filename):
        X = np.loadtxt(self, filename)
        return X

    def encoding_latency_MNIST(self, coding_f, analog_data, coding_n, min=0, max=np.pi):
        pass

    def get_series_data(self, data_frame, is_group=False):
        pass

    def get_series_data_list(self, data_frame, is_group=False):
        pass

    def select_data(self, fraction, data_frame, is_order=True, **kwargs):
        pass


# X_train = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/train/X_train.txt')
# y_train = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/train/y_train.txt')
# X_test = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/test/X_test.txt')
# y_test = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/test/y_test.txt')