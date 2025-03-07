# -*- coding: utf-8 -*-
"""
    The functions for preparing the data.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions

import os
import shutil, zipfile
from PIL import Image as PIL_Image
import cv2 as cv

import numpy as np
import pandas as pd


class BN_classification(BaseFunctions):
    """
    Class used to load BN dataset

    """
    def __init__(self):
        self.IMG_HEIGHT = 100
        self.IMG_WIDTH = 150
        self.NUM_FRAMES = 20

    def _get_category(self, path_category):
        """Loads the Dataframe labels from a csv and creates dictionnaries to convert the string labels to int and backwards
        # Arguments
            path_labels : path to the csv containing the labels
        """
        self.category_df = pd.read_csv(path_category, names=['category'])
        # Extracting list of labels from the dataframe
        self.category = [str(category[0]) for category in self.category_df.values]
        self.n_category = len(self.category)
        # Create dictionnaries to convert label to int and backwards
        self.CATEGORIES = dict(zip(self.category, range(self.n_category)))
        self.label_to_category = dict(enumerate(self.category))

    def _load_video_category(self, path_subset, mode="category"):
        """ Loads a Dataframe from a csv
        # Arguments
            path_subset : String, path to the csv to load
            mode        : String, (default: label), if mode is set to "label", filters rows given if the labels exists in the labels Dataframe loaded previously
        #Returns
            A DataFrame
        """
        if mode == "input":
            names = ['video_id']
        elif mode == "category":
            names = ['video_id', 'category']
        df = pd.read_csv(path_subset, sep=';', names=names)
        if mode == "category":
            df = df[df.category.isin(self.category)]
        return df

    def load_labels(self):
        self._get_category(self.path_labels)
        if self.path_train:
            self.train = self._load_video_category(self.path_train)
        if self.path_validation:
            self.validation = self._load_video_category(self.path_validation)
        if self.path_test:
            self.test = self._load_video_category(self.path_test, 'input')

    def normalize2(self, number):
        stre = str(number)
        while len(stre) < 5:
            stre = "0" + stre
        return stre

    def extract(self, z, sub_path):
        i = 1
        while True:
            try:
                z.extract(sub_path + self.normalize2(i) + '.jpg', path=self.data_path)
                i += 1
            except KeyError:
                break

    def delete(self, sub_path):
        shutil.rmtree(sub_path)

    def load_videos(self, videoIds, z):
        video_set = []
        for counterVideo, videoId in enumerate(videoIds, 1):
            self.extract(z, str(videoId) + '/')
            directory = os.path.join(self.data_path, str(videoId) + '/')
            if len(os.listdir(directory)) >= self.NUM_FRAMES:  # don't use if video length is too small
                video = []
                for counterImage, image in enumerate(os.listdir(directory), 1):
                    image_values = PIL_Image.open(os.path.join(directory, image))
                    image_values = image_values.convert('RGB')  # L: converts to greyscale
                    image_values = image_values.resize((self.IMG_WIDTH, self.IMG_HEIGHT), PIL_Image.ANTIALIAS)
                    video.append(np.asarray(image_values).reshape(-1))
                video_set.append(np.asarray(video))
            else:
                # print("skipped video # %d with length %d" % (counterVideo, len(os.listdir(directory))))
                video_set.append(None)
            self.delete(directory)
        return np.asarray(video_set)

    def optical_flow(self, frames, origin_size=(100, 150, 3)):
        frame1 = frames[0].reshape(origin_size)
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        bgrs = []
        for frame_ in frames[1:]:
            frame2 = frame_.reshape(origin_size)
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            # 返回一个两通道的光流向量，实际上是每个点的像素位移值
            flow = cv.calcOpticalFlowFarneback(prvs, next, None,
                                               0.5, 3, 5,
                                               3, 5, 1.1, 0)
            # 笛卡尔坐标转换为极坐标，获得极轴和极角
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

            hsv[..., 0] = ang * 180 / np.pi / 2  # 角度
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            bgrs.append(bgr)
            prvs = next
        return bgrs

    def load_data_BN(self, videoId_df):
        frames = self.load_videos(videoId_df.video_id.values, self.z)
        videoId_df_ = videoId_df.copy()
        videoId_df_['frames'] = frames
        return videoId_df_.dropna(axis=0, subset=['frames'])

    def load_data_BN_all(self, data_path):
        self.data_path = data_path
        self.path_vid = data_path + '20bn-jester-v1.zip'
        self.path_labels = data_path + 'jester-v1-labels.csv'
        self.path_train = data_path + 'jester-v1-train.csv'
        self.path_validation = data_path + 'jester-v1-validation.csv'
        self.path_test = data_path + 'jester-v1-test.csv'
        self.load_labels()
        self.z = zipfile.ZipFile(self.path_vid, 'r', zipfile.ZIP_DEFLATED)

    def frame_diff(self, frames, origin_size=(100, 150, 3), with_optical_flow = True):
        frame_diff = []
        it = frames.__iter__()
        frame_pre = cv.cvtColor(next(it).reshape(origin_size), cv.COLOR_BGR2GRAY)
        while True:
            try:
                frame = cv.cvtColor(next(it).reshape(origin_size), cv.COLOR_BGR2GRAY)
                frame_diff.append(np.abs(frame_pre.astype(int) - frame.astype(int)).reshape(-1))
                frame_pre = frame
            except StopIteration:
                break
        if with_optical_flow:
            bgrs = self.optical_flow(frames, origin_size)
            frame_diff_masked = []
            for diff, bgr in zip(frame_diff, bgrs):
                masked = diff.reshape(origin_size[:2]) * (cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(int)/255)
                frame_diff_masked.append(masked)
            return np.asarray(frame_diff_masked)
        else:
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

    def pooling(self, frames, origin_size=(100, 150), pool_size=(5, 5), types='max'):
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

    def select_data_BN(self, fraction, data_frame, is_order=True, **kwargs):
        try:
            selected = [self.label_to_category[x] for x in kwargs['selected']]
        except KeyError:
            selected = self.CATEGORIES.keys()
        if  fraction > 1:
            data_frame_selected = pd.DataFrame(columns=data_frame.columns)
            for s in selected:
                data_frame_selected = data_frame_selected.append(data_frame[data_frame['category'].isin([s])].sample(
                    n=fraction))
        else:
            data_frame_selected = data_frame[data_frame['category'].isin(selected)].sample(frac=fraction)

        if is_order:
            data_frame_selected = data_frame_selected.sort_index().reset_index(drop=True)
        else:
            data_frame_selected = data_frame_selected.reset_index(drop=True)
        return self.load_data_BN(data_frame_selected)

    def encoding_latency_BN(self, analog_data, origin_size=(100, 150, 3), pool_size=(5, 5), types='max', threshold=0.2,
                            with_optical_flow=True):
        data_diff = analog_data.frames.apply(self.frame_diff, origin_size=origin_size, with_optical_flow = with_optical_flow)
        data_diff_pool = data_diff.apply(self.pooling, origin_size=origin_size[:2], pool_size=pool_size, types=types)
        data_diff_pool_threshold_norm = data_diff_pool.apply(self.threshold_norm, threshold=threshold)
        label = analog_data.category.map(self.CATEGORIES).astype('<i1')
        data_frame = pd.DataFrame({'value': data_diff_pool_threshold_norm, 'label': label})
        return data_frame