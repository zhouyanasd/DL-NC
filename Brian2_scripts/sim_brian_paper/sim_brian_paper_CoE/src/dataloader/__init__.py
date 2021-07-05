# -*- coding: utf-8 -*-
"""
    The dataloader methods used for CoE.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from .KTH_dataloader import *
from .UCI_phone_dataloader import *

__all__ = [
    "KTH_classification",
    "UCI_classification",
]

# def Dataloader(data_type, data_path, DataName, F_train, F_pre_train, F_validation, core,
#                origin_size, pool_size, pool_types, pool_threshold):
#     if data_type == 'KTH':
#         DATA = KTH_classification()
#         original_data = data_path
#         saved_path = data_path
#     elif data_type == 'UCI':
#         DATA = UCI_classification()
#         original_data = data_path + 'Raw-Data/'
#         saved_path = data_path + 'Spike-train-Data/'
#
#     try:
#         df_en_train = DATA.load_data(saved_path + 'train_' + DataName + '.p')
#         df_en_pre_train = DATA.load_data(saved_path + 'pre_train_' + DataName + '.p')
#         df_en_validation = DATA.load_data(saved_path + 'validation_' + DataName + '.p')
#         df_en_test = DATA.load_data(saved_path + 'test_' + DataName + '.p')
#     except FileNotFoundError:
#         DATA.load_data_DATA_all(original_data, split_type='mixed', split=[15, 5, 4])
#
#         df_train = DATA.select_data(F_train, DATA.train, False)
#         df_pre_train = DATA.select_data(F_pre_train, DATA.train, False)
#         df_validation = DATA.select_data(F_validation, DATA.validation, False)
#         df_test = DATA.select_data(F_train, DATA.test, False)
#
#         df_en_train = DATA.encoding_latency(df_train, origin_size, pool_size, pool_types, pool_threshold)
#         df_en_pre_train = DATA.encoding_latency(df_pre_train, origin_size, pool_size, pool_types, pool_threshold)
#         df_en_validation = DATA.encoding_latency(df_validation, origin_size, pool_size, pool_types, pool_threshold)
#         df_en_test = DATA.encoding_latency(df_test, origin_size, pool_size, pool_types, pool_threshold)
#
#         DATA.dump_data(saved_path + 'train_' + DataName + '.p', df_en_train)
#         DATA.dump_data(saved_path + 'pre_train_' + DataName + '.p', df_en_pre_train)
#         DATA.dump_data(saved_path + 'validation_' + DataName + '.p', df_en_validation)
#         DATA.dump_data(saved_path + 'test_' + DataName + '.p', df_en_test)
#
#     data_train_index_batch = DATA.data_batch(df_en_train.index.values, core)
#     data_pre_train_index_batch = DATA.data_batch(df_en_pre_train.index.values, core)
#     data_validation_index_batch = DATA.data_batch(df_en_validation.index.values, core)
#     data_test_index_batch = DATA.data_batch(df_en_test.index.values, core)