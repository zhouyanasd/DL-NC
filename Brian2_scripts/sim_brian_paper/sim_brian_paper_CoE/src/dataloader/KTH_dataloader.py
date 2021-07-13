# -*- coding: utf-8 -*-
"""
    The functions for preparing the data.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.dataloader import Dataloader

import os
import re

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class KTH_classification(BaseFunctions, Dataloader):
    def __init__(self):
        self.CATEGORIES = {
            "boxing": 0,
            "handclapping": 1,
            "handwaving": 2,
            "jogging": 3,
            "running": 4,
            "walking": 5
        }

    def spilt_data(self, split_type, **kwargs):
        if split_type == 'official':
            self.TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
            self.VALIDATION_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
            self.TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]
        elif split_type == 'random':
            x = np.arange(25) + 1
            np.random.shuffle(x)
            s = kwargs['split']
            self.TRAIN_PEOPLE_ID = x[:s[0]]
            self.VALIDATION_PEOPLE_ID = x[s[0]:sum(s[:2])]
            self.TEST_PEOPLE_ID = x[sum(s[:2]):sum(s)]
        elif split_type == 'solid':
            self.TRAIN_PEOPLE_ID = kwargs['train']
            self.VALIDATION_PEOPLE_ID = kwargs['validation']
            self.TEST_PEOPLE_ID = kwargs['test']
        elif split_type == 'mixed':
            self.TRAIN_PEOPLE_ID = np.arange(25) + 1
        else:
            print('worng type, use official instead')
            self.TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
            self.VALIDATION_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
            self.TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

    def parse_sequence_file(self, path):
        print("Parsing %s" % path)

        with open(path, 'r') as content_file:
            content = content_file.read()
        content = re.sub("[\t\n]", " ", content).split()
        self.frames_idx = {}
        current_filename = ""
        for s in content:
            if s == "frames":
                continue
            elif s.find("-") >= 0:
                if s[len(s) - 1] == ',':
                    s = s[:-1]
                idx = s.split("-")
                if current_filename[:6] == 'person':
                    if not current_filename in self.frames_idx:
                        self.frames_idx[current_filename] = []
                    self.frames_idx[current_filename].append([int(idx[0]), int(idx[1])])
            else:
                current_filename = s + "_uncomp.avi"

    def load_data_KTH(self, data_path, dataset="train"):
        if dataset == "train":
            ID = self.TRAIN_PEOPLE_ID
        elif dataset == "validation":
            ID = self.VALIDATION_PEOPLE_ID
        else:
            ID = self.TEST_PEOPLE_ID

        data = []
        for category in self.CATEGORIES.keys():
            folder_path = os.path.join(data_path, category)
            filenames = sorted(os.listdir(folder_path))
            for filename in filenames:
                filepath = os.path.join(data_path, category, filename)
                person_id = int(filename.split("_")[0][6:])
                if person_id not in ID:
                    continue
                condition_id = int(filename.split("_")[2][1:])
                cap = cv2.VideoCapture(filepath)
                for f_id, seg in enumerate(self.frames_idx[filename]):
                    frames = []
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seg[0] - 1)  # 设置要获取的帧号
                    count = 0
                    while (cap.isOpened() and seg[0] + count - 1 < seg[1] + 1):
                        ret, frame = cap.read()
                        if ret == True:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frames.append(gray.reshape(-1))
                            count += 1
                        else:
                            break
                    data.append({
                        "frames": np.array(frames),
                        "category": category,
                        "person_id": person_id,
                        "condition_id": condition_id,
                        "frame_id": f_id,
                    })
                cap.release()
        return pd.DataFrame(data)

    def frame_diff(self, frames, origin_size=(120, 160)):
        frame_diff = []
        it = frames.__iter__()
        frame_pre = next(it).reshape(origin_size)
        while True:
            try:
                frame = next(it).reshape(origin_size)
                frame_diff.append(cv2.absdiff(frame_pre, frame).reshape(-1))
                frame_pre = frame
            except StopIteration:
                break
        return np.asarray(frame_diff)

    def block_array(self, matrix, size):
        if int(matrix.shape[0] % size[0]) == 0 and int(matrix.shape[1] % size[1]) == 0:
            X = int(matrix.shape[0] / size[0])
            Y = int(matrix.shape[1] / size[1])
            shape = (X, Y, size[0], size[1])
            strides = matrix.itemsize * np.array([matrix.shape[1] * size[0], size[1], matrix.shape[1], 1])
            squares = np.lib.stride_tricks.as_strided(matrix, shape=shape, strides=strides)
            return squares
        else:
            raise ValueError('matrix must be divided by size exactly')

    def pooling(self, frames, origin_size=(120, 160), pool_size=(5, 5), types='max'):
        data = []
        for frame in frames:
            pool = np.zeros((int(origin_size[0] / pool_size[0]), int(origin_size[1] / pool_size[1])), dtype=np.int16)
            frame_block = self.block_array(frame.reshape(origin_size), pool_size)
            for i, row in enumerate(frame_block):
                for j, block in enumerate(row):
                    if types == 'max':
                        pool[i][j] = block.max()
                    elif types == 'average':
                        pool[i][j] = block.mean()
                    else:
                        raise ValueError('I have not done that type yet..')
            data.append(pool.reshape(-1))
        return np.asarray(data)

    def threshold_norm(self, frames, threshold):
        frames = (frames - np.min(frames)) / (np.max(frames) - np.min(frames))
        frames[frames < threshold] = 0
        frames[frames > threshold] = 1
        frames = frames.astype('<i1')
        return frames

    def load_data_KTH_all(self, data_path, split_type, **kwargs):
        self.spilt_data(split_type, **kwargs)
        self.parse_sequence_file(data_path + '00sequences.txt')
        if split_type == 'mixed':
            self.train = self.load_data_KTH(data_path, dataset="train")
            self.train = self.select_data_KTH(1, self.train, False)
            self.train, self.test = train_test_split(self.train,
                                                     test_size=kwargs['split'][-1] / sum(kwargs['split']),
                                                     random_state=42)
            self.train, self.validation = train_test_split(self.train,
                                                           test_size=kwargs['split'][1] / sum(kwargs['split'][:2]),
                                                           random_state=42)
        else:
            self.train = self.load_data_KTH(data_path, dataset="train")
            self.validation = self.load_data_KTH(data_path, dataset="validation")
            self.test = self.load_data_KTH(data_path, dataset="test")

    def select_data_KTH(self, fraction, data_frame, is_order=True, **kwargs):
        try:
            selected = kwargs['selected']
        except KeyError:
            selected = self.CATEGORIES.keys()
        if is_order:
            data_frame_selected = data_frame[data_frame['category'].isin(selected)].sample(
                frac=fraction).sort_index().reset_index(drop=True)
        else:
            data_frame_selected = data_frame[data_frame['category'].isin(selected)].sample(frac=fraction).reset_index(
                drop=True)
        return data_frame_selected

    def encoding_latency_KTH(self, analog_data, origin_size=(120, 160), pool_size=(5, 5), types='max', threshold=0.2):
        data_diff = analog_data.frames.apply(self.frame_diff, origin_size=origin_size)
        data_diff_pool = data_diff.apply(self.pooling, origin_size=origin_size, pool_size=pool_size, types=types)
        data_diff_pool_threshold_norm = data_diff_pool.apply(self.threshold_norm, threshold=threshold)
        label = analog_data.category.map(self.CATEGORIES).astype('<i1')
        data_frame = pd.DataFrame({'value': data_diff_pool_threshold_norm, 'label': label})
        return data_frame