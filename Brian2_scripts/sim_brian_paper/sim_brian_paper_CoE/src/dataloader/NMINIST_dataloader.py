# -*- coding: utf-8 -*-
"""
    The functions for preparing the data.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions

import shutil, zipfile
import numpy as np
import pandas as pd


class NMINIST_classification(BaseFunctions):
    """
    Class used to load NMINIST dataset

    """
    def __init__(self, shape = (34, 34), duration = 10):
        self.shape = shape
        self.duration = duration

    def load_ATIS_bin(self, file_name: str) -> dict:
        '''
        :param file_name: path of the aedat v3 file
        :type file_name: str
        :return: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
        :rtype: Dict
        This function is written by referring to https://github.com/jackd/events-tfds .
        Each ATIS binary example is a separate binary file consisting of a list of events. Each event occupies 40 bits as described below:
        bit 39 - 32: Xaddress (in pixels)
        bit 31 - 24: Yaddress (in pixels)
        bit 23: Polarity (0 for OFF, 1 for ON)
        bit 22 - 0: Timestamp (in microseconds)
        '''
        with open(file_name, 'rb') as bin_f:
            # `& 128` 是取一个8位二进制数的最高位
            # `& 127` 是取其除了最高位，也就是剩下的7位
            raw_data = np.uint32(np.fromfile(bin_f, dtype=np.uint8))
            x = raw_data[0::5]
            y = raw_data[1::5]
            rd_2__5 = raw_data[2::5]
            p = (rd_2__5 & 128) >> 7
            t = ((rd_2__5 & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        return {'t': t, 'x': x, 'y': y, 'p': p}

    def integrate_events_segment_to_frame(self, events: dict, H: int, W: int, j_l: int = 0, j_r: int = -1) -> np.ndarray:
        '''
        :param events: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
        :type events: Dict
        :param H: height of the frame
        :type H: int
        :param W: weight of the frame
        :type W: int
        :param j_l: the start index of the integral interval, which is included
        :type j_l: int
        :param j_r: the right index of the integral interval, which is not included
        :type j_r:
        :return: frames
        :rtype: np.ndarray
        Denote a two channels frame as :math:`F` and a pixel at :math:`(p, x, y)` as :math:`F(p, x, y)`, the pixel value is integrated from the events data whose indices are in :math:`[j_{l}, j_{r})`:
    .. math::
        F(p, x, y) &= \sum_{i = j_{l}}^{j_{r} - 1} \mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})
    where :math:`\lfloor \cdot \rfloor` is the floor operation, :math:`\mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})` is an indicator function and it equals 1 only when :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})`.
        '''
        # 累计脉冲需要用bitcount而不能直接相加，原因可参考下面的示例代码，以及
        # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
        # We must use ``bincount`` rather than simply ``+``. See the following reference:
        # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments

        # Here is an example:

        # height = 3
        # width = 3
        # frames = np.zeros(shape=[2, height, width])
        # events = {
        #     'x': np.asarray([1, 2, 1, 1]),
        #     'y': np.asarray([1, 1, 1, 2]),
        #     'p': np.asarray([0, 1, 0, 1])
        # }
        #
        # frames[0, events['y'], events['x']] += (1 - events['p'])
        # frames[1, events['y'], events['x']] += events['p']
        # print('wrong accumulation\n', frames)
        #
        # frames = np.zeros(shape=[2, height, width])
        # for i in range(events['p'].__len__()):
        #     frames[events['p'][i], events['y'][i], events['x'][i]] += 1
        # print('correct accumulation\n', frames)
        #
        # frames = np.zeros(shape=[2, height, width])
        # frames = frames.reshape(2, -1)
        #
        # mask = [events['p'] == 0]
        # mask.append(np.logical_not(mask[0]))
        # for i in range(2):
        #     position = events['y'][mask[i]] * width + events['x'][mask[i]]
        #     events_number_per_pos = np.bincount(position)
        #     idx = np.arange(events_number_per_pos.size)
        #     frames[i][idx] += events_number_per_pos
        # frames = frames.reshape(2, height, width)
        # print('correct accumulation by bincount\n', frames)

        frame = np.zeros(shape=[2, H * W])
        x = events['x'][j_l: j_r].astype(int)  # avoid overflow
        y = events['y'][j_l: j_r].astype(int)
        p = events['p'][j_l: j_r]
        mask = []
        mask.append(p == 0)
        mask.append(np.logical_not(mask[0]))
        for c in range(2):
            position = y[mask[c]] * W + x[mask[c]]
            events_number_per_pos = np.bincount(position)
            frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
        return frame.reshape((2, H, W))

    def cal_fixed_frames_number_segment_index(self, events_t: np.ndarray, split_by: str, frames_num: int) -> tuple:
        '''
        :param events_t: events' t
        :type events_t: numpy.ndarray
        :param split_by: 'time' or 'number'
        :type split_by: str
        :param frames_num: the number of frames
        :type frames_num: int
        :return: a tuple ``(j_l, j_r)``
        :rtype: tuple
        Denote ``frames_num`` as :math:`M`, if ``split_by`` is ``'time'``, then
        .. math::
            \\Delta T & = [\\frac{t_{N-1} - t_{0}}{M}] \\\\
            j_{l} & = \\mathop{\\arg\\min}\\limits_{k} \\{t_{k} | t_{k} \\geq t_{0} + \\Delta T \\cdot j\\} \\\\
            j_{r} & = \\begin{cases} \\mathop{\\arg\\max}\\limits_{k} \\{t_{k} | t_{k} < t_{0} + \\Delta T \\cdot (j + 1)\\} + 1, & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}
        If ``split_by`` is ``'number'``, then
        .. math::
            j_{l} & = [\\frac{N}{M}] \\cdot j \\\\
            j_{r} & = \\begin{cases} [\\frac{N}{M}] \\cdot (j + 1), & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}
        '''
        j_l = np.zeros(shape=[frames_num], dtype=int)
        j_r = np.zeros(shape=[frames_num], dtype=int)
        N = events_t.size

        if split_by == 'number':
            di = N // frames_num
            for i in range(frames_num):
                j_l[i] = i * di
                j_r[i] = j_l[i] + di
            j_r[-1] = N

        elif split_by == 'time':
            dt = (events_t[-1] - events_t[0]) // frames_num
            idx = np.arange(N)
            for i in range(frames_num):
                t_l = dt * i + events_t[0]
                t_r = t_l + dt
                mask = np.logical_and(events_t >= t_l, events_t < t_r)
                idx_masked = idx[mask]
                j_l[i] = idx_masked[0]
                j_r[i] = idx_masked[-1] + 1

            j_r[-1] = N
        else:
            raise NotImplementedError

        return j_l, j_r

    def integrate_events_by_fixed_frames_number(self, events: dict, split_by: str, frames_num: int, H: int,
                                                W: int) -> np.ndarray:
        '''
        :param events: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
        :type events: Dict
        :param split_by: 'time' or 'number'
        :type split_by: str
        :param frames_num: the number of frames
        :type frames_num: int
        :param H: the height of frame
        :type H: int
        :param W: the weight of frame
        :type W: int
        :return: frames
        :rtype: np.ndarray
        Integrate events to frames by fixed frames number. See ``cal_fixed_frames_number_segment_index`` and ``integrate_events_segment_to_frame`` for more details.
        '''
        j_l, j_r = self.cal_fixed_frames_number_segment_index(events['t'], split_by, frames_num)
        frames = np.zeros([frames_num, 2, H, W])
        for i in range(frames_num):
            frames[i] = self.integrate_events_segment_to_frame(events, H, W, j_l[i], j_r[i])
        return frames

    def load_labels(self, z):
        label = []
        name_list = []
        for name in z.namelist():
            if len(name) > 8:
                label.append(int(name[-11]))
                name_list.append(name[-9:])
        return pd.DataFrame({'name': name_list, 'label': label})

    def load_data_NMINIST(self, data_df):
        data_list = []
        if data_df.name[0] in self.train.name.values:
            path = 'Train/'
            z = self.z_train
        elif data_df.name[0] in self.test.name.values:
            path = 'Test/'
            z = self.z_test
        for label, data_name in zip(data_df.label.values, data_df.name.values):
            path_file = path+ str(label) +'/'+ data_name
            z.extract(path_file, path=self.data_path)
            data = self.integrate_events_by_fixed_frames_number(self.load_ATIS_bin(
                self.data_path + path_file), 'time', self.duration, self.shape[0], self.shape[1]).reshape(self.duration, -1)
            data_list.append(data)
        shutil.rmtree(self.data_path+path)
        data_df_ = data_df.copy()
        data_df_['frames'] = data_list
        return data_df_

    def load_data_NMINIST_all(self, data_path):
        self.data_path = data_path
        self.path_train = data_path + 'Train.zip'
        self.path_test = data_path + 'Test.zip'
        self.z_train = zipfile.ZipFile(self.path_train, 'r', zipfile.ZIP_DEFLATED)
        self.z_test = zipfile.ZipFile(self.path_test, 'r', zipfile.ZIP_DEFLATED)
        self.train = self.load_labels(self.z_train)
        self.test = self.load_labels(self.z_test)

    def threshold(self, frames, threshold):
        frames = (frames>threshold).astype('<i1')
        return frames

    def select_data_NMINIST(self, fraction, data_frame, is_order=True, **kwargs):
        try:
            selected = kwargs['selected']
        except KeyError:
            selected = list(np.arange(9))
        if  fraction > 1:
            data_frame_selected = pd.DataFrame(columns=data_frame.columns)
            for s in selected:
                data_frame_selected = data_frame_selected.append(data_frame[data_frame['label'].isin([s])].sample(
                    n=fraction))
        else:
            data_frame_selected = data_frame[data_frame['label'].isin(selected)].sample(frac=fraction)

        if is_order:
            data_frame_selected = data_frame_selected.sort_index().reset_index(drop=True)
        else:
            data_frame_selected = data_frame_selected.reset_index(drop=True)
        return self.load_data_NMINIST(data_frame_selected)

    def encoding_latency_NMINIST(self, analog_data, threshold=0):
        data_threshold = analog_data.frames.apply(self.threshold, threshold=threshold)
        label = analog_data.label.values.astype('<i1')
        data_frame = pd.DataFrame({'value': data_threshold, 'label': label})
        return data_frame