# -*- coding: utf-8 -*-
"""
    The functions for preparing the data.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

# from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions

import os
import pickle, shutil, zipfile
from PIL import Image as PIL_Image

import numpy as np
import pandas as pd

class BN_classification():
    """
    Class used to load BN dataset

    """
    def __init__(self):
        self.IMG_HEIGHT = 100
        self.IMG_WIDTH = 150
        self.NUM_FRAMES = 30

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
                    image_values = image_values.convert('L')  # L: converts to greyscale
                    image_values = image_values.resize((self.IMG_WIDTH, self.IMG_HEIGHT), PIL_Image.ANTIALIAS)
                    video.append(np.asarray(image_values).reshape(-1))
                video_set.append(np.asarray(video))
            else:
                # print("skipped video # %d with length %d" % (counterVideo, len(os.listdir(directory))))
                video_set.append(None)
            self.delete(directory)
        return np.asarray(video_set)

    def load_data_BN(self, videoId_df):
        z = zipfile.ZipFile(self.path_vid, 'r', zipfile.ZIP_DEFLATED)
        frames = self.load_videos(videoId_df.video_id.values, z)
        z.close()
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

    def frame_diff(self, frames, origin_size=(150, 100)):
        frame_diff = []
        it = frames.__iter__()
        frame_pre = next(it).reshape(origin_size)
        while True:
            try:
                frame = next(it).reshape(origin_size)
                frame_diff.append(np.abs(frame_pre.astype(int) - frame.astype(int)).reshape(-1))
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

    def pooling(self, frames, origin_size=(150, 100), pool_size=(5, 5), types='max'):
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
        if is_order:
            data_frame_selected = data_frame[data_frame['category'].isin(selected)].sample(
                frac=fraction).sort_index().reset_index(drop=True)
        else:
            data_frame_selected = data_frame[data_frame['category'].isin(selected)].sample(frac=fraction).reset_index(
                drop=True)
        return self.load_data_BN(data_frame_selected)

    def encoding_latency_BN(self, analog_data, origin_size=(150, 100), pool_size=(5, 5), types='max', threshold=0.2):
        data_diff = analog_data.frames.apply(self.frame_diff, origin_size=origin_size)
        data_diff_pool = data_diff.apply(self.pooling, origin_size=origin_size, pool_size=pool_size, types=types)
        data_diff_pool_threshold_norm = data_diff_pool.apply(self.threshold_norm, threshold=threshold)
        label = analog_data.category.map(self.CATEGORIES).astype('<i1')
        data_frame = pd.DataFrame({'value': data_diff_pool_threshold_norm, 'label': label})
        return data_frame

    def get_series_data_list(self, data_frame, is_group=False):
        data_frame_s = []
        if not is_group:
            for value in data_frame['value']:
                data_frame_s.extend(value)
        else:
            for value in data_frame['value']:
                data_frame_s.append(value)
        label = data_frame['label']
        return np.asarray(data_frame_s), label

    def dump_data(self, path, dataset):
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'wb') as file:
            pickle.dump(dataset, file)

    def load_data(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

if __name__ == "__main__":
    DataName = 'coe_0.01'
    data_path = "C:/Users/Administrator/Desktop/share/"
    BN = BN_classification()
    BN.load_data_BN_all(data_path)

    df_train = BN.select_data_BN(0.01, BN.train, False)
    df_pre_train = BN.select_data_BN(0.001, BN.train, False)
    df_validation = BN.select_data_BN(0.01, BN.validation, False)
    df_test = BN.select_data_BN(0.01, BN.validation, False)

    df_en_train = BN.encoding_latency_BN(df_train)
    df_en_pre_train = BN.encoding_latency_BN(df_pre_train)
    df_en_validation = BN.encoding_latency_BN(df_validation)
    df_en_test = BN.encoding_latency_BN(df_test)

    BN.dump_data(data_path + 'train_' + DataName + '.p', df_en_train)
    BN.dump_data(data_path + 'pre_train_' + DataName + '.p', df_en_pre_train)
    BN.dump_data(data_path + 'validation_' + DataName + '.p', df_en_validation)
    BN.dump_data(data_path + 'test_' + DataName + '.p', df_en_test)