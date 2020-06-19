import os
import logging
import numpy as np
import pandas as pd
import imageio as io

from tqdm import tqdm


class JesterDataSet:
    N_CLASSES = 27
    DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', '20BN-JESTER')

    def __init__(self, partition='train', classes=None, proportion=1.0, video_shape=(30, 100, 100, 3), data_shape=None,
                 batch_size=10, cutoff=True, preload=True, dtype=np.float16, verbose=False, data_path=None):
        """
        Container for the 20BN-JESTER dataset. Note that both the data and the labels have to be downloaded manually
        into the directory specified as an argument. The data can be found at https://www.twentybn.com/datasets/jester.

        Arguments:
            partition: value in {'train', 'validation', 'test'}
            classes: classes that should be loaded. Either None, in which case all of the classes will be loaded,
                an int n, in which case n first classes will be loaded, or a list of ints representing IDs, in which
                case the classes with the specified IDs will be loaded
            proportion: proportion of the data that should be loaded for each class
            video_shape: dimensions of the input videos, in a format (n_frames, height, width, n_channels)
            cutoff: True if the number of the videos should be a multiplier of the batch size (remaining images will
                be discarded), False otherwise
            preload: True if the whole dataset should be preloaded to the memory, False if each batch should be loaded
                from the hard drive when required
            verbose: True if the progress bar for loading the videos should be displayed, False otherwise
            data_path: directory containing the data, falls back to the default value if None
        """
        assert partition in ['train', 'validation', 'test']
        assert classes is None \
               or (type(classes) is int and classes <= self.N_CLASSES) \
               or (min(classes) >= 0 and max(classes) < self.N_CLASSES)
        assert 0 < proportion <= 1.0

        self.partition = partition
        self.video_shape = video_shape
        self.batch_size = batch_size
        self.preload = preload
        self.dtype = dtype
        self.verbose = verbose
        self.data_path = data_path
        self.label_names = []
        self.label_dictionary = {}
        self.labels = None
        self.video_ids = []
        self.videos = None
        self.videos_completed = 0
        self.length = 0

        if classes is None:
            classes = range(0, 27)
        elif type(classes) is int:
            classes = range(0, classes)

        if data_shape is None:
            self.data_shape = self.video_shape
        else:
            self.data_shape = data_shape

        if self.data_path is None:
            self.data_path = self.DEFAULT_DATA_PATH

        assert os.path.exists(self.data_path)

        df = pd.read_csv(os.path.join(self.data_path, 'jester-v1-labels.csv'), header=None)

        self.label_names = list(df.loc[classes, 0])

        for cls in range(len(classes)):
            self.label_dictionary[self.label_names[cls]] = cls

        df = pd.read_csv(os.path.join(self.data_path, 'jester-v1-%s.csv' % partition), header=None, delimiter=';')

        if partition in ['train', 'validation']:
            grouped_video_ids = {}

            for cls in range(len(classes)):
                grouped_video_ids[cls] = list(df[df[1] == self.label_names[cls]][0])

            if proportion < 1.0:
                for cls in range(len(classes)):
                    n = int(proportion * len(grouped_video_ids[cls]))

                    assert n > 0

                    grouped_video_ids[cls] = grouped_video_ids[cls][:n]

            self.video_ids = np.array(sum(grouped_video_ids.values(), []))
            self.labels = np.array(sum([[cls] * len(grouped_video_ids[cls]) for cls in range(len(classes))], []))
        else:
            self.video_ids = np.array(df[0])

        self.length = len(self.video_ids)

        if cutoff:
            self.length = self.length - self.length % batch_size

            self.video_ids = self.video_ids[:self.length]

            if self.labels is not None:
                self.labels = self.labels[:self.length]

        if preload:
            self.videos = self.load_videos(self.video_ids)

    def shuffle(self, indices=None):
        if indices is None:
            indices = list(range(len(self.video_ids)))
            np.random.shuffle(indices)

        self.video_ids = self.video_ids[indices]

        if self.labels is not None:
            self.labels = self.labels[indices]

        if self.videos is not None:
            self.videos = self.videos[indices]

    def load_video(self, video_id):
        video_directory = os.path.join(self.data_path, '20bn-jester-v1', str(video_id))

        assert os.path.exists(video_directory)

        frame_names = sorted([fn for fn in os.listdir(video_directory) if fn.endswith('.jpg')])
        n_frames = len(frame_names)

        if n_frames >= self.video_shape[0]:
            first_frame_index = int((n_frames - self.video_shape[0]) / 2)
            frame_names = frame_names[first_frame_index:(first_frame_index + self.video_shape[0])]
        else:
            n_frames_missing = self.video_shape[0] - n_frames
            n_before = int(n_frames_missing / 2)
            n_after = n_frames_missing - n_before
            frame_names = [frame_names[0]] * n_before + frame_names + [frame_names[-1]] * n_after

        frames = []

        for frame_name in frame_names:
            frames.append(np.array(io.imread(os.path.join(video_directory, frame_name)), dtype=self.dtype))

        video = np.stack(frames)

        assert video.shape[1] == self.video_shape[1]
        assert video.shape[3] == self.video_shape[3]
        assert video.shape[2] >= self.video_shape[2]

        width_start = int((video.shape[2] - self.video_shape[2]) / 2)
        video = video[:, :, width_start:(width_start + self.video_shape[2]), :]

        return video

    def load_videos(self, video_ids):
        iterator = range(len(video_ids))

        if self.preload and self.verbose:
            logging.info('Loading the %s partition of the Jester dataset...' % self.partition)

            iterator = tqdm(iterator)

        videos = np.zeros([len(video_ids)] + list(self.data_shape), dtype=self.dtype)

        for i in iterator:
            video_id = video_ids[i]
            videos[i] = self.load_video(video_id)

        return videos

    def batch(self):
        if self.videos is None:
            videos = self.load_videos(self.video_ids[self.videos_completed:(self.videos_completed + self.batch_size)])
        else:
            videos = self.videos[self.videos_completed:(self.videos_completed + self.batch_size)]

        labels = self.labels[self.videos_completed:(self.videos_completed + self.batch_size)]

        self.videos_completed += self.batch_size

        if self.videos_completed >= self.length:
            self.videos_completed = 0

        return videos, labels


class FlattenedJesterDataSet(JesterDataSet):
    def __init__(self, axis, **kwargs):
        self.axis = axis

        kwargs['data_shape'] = (kwargs['video_shape'][axis],
                                np.prod([kwargs['video_shape'][i] for i in range(3) if i != axis]),
                                kwargs['video_shape'][-1])

        super().__init__(**kwargs)

    def load_video(self, video_id):
        video = super().load_video(video_id)

        return self._flatten(video, self.axis)

    @staticmethod
    def _flatten(tensor, axis):
        assert len(tensor.shape) == 4

        flattened_tensor = []

        for i in range(tensor.shape[3]):
            channel = tensor[:, :, :, i]

            # roll axis order so that the first tensor dimension is the specified axis
            channel = np.transpose(channel, (axis, (axis + 1) % 3, (axis + 2) % 3))

            flattened_channel = np.zeros((channel.shape[0], channel.shape[1] * channel.shape[2]), dtype=channel.dtype)
            position = 0

            for k in range(channel.shape[2]):
                for j in range(channel.shape[1]):
                    flattened_channel[:, position] = channel[:, j, k]
                    position += 1

            flattened_tensor.append(flattened_channel)

        return np.moveaxis(np.array(flattened_tensor), 0, -1)


class MultiStreamJesterDataSet:
    N_CLASSES = JesterDataSet.N_CLASSES

    def __init__(self, stream_types=('C3D', 'C2D_0', 'C2D_1', 'C2D_2'), **kwargs):
        self.datasets = []
        self.data_shape = []

        for stream_type in stream_types:
            if stream_type.startswith('C3D'):
                self.datasets.append(JesterDataSet(**kwargs))
            elif stream_type.startswith('C2D_'):
                axis = int(stream_type[4:])

                self.datasets.append(FlattenedJesterDataSet(axis=axis, **kwargs))
            else:
                raise ValueError

            self.data_shape.append(self.datasets[-1].data_shape)

        self.flattened_shape = np.prod(self.datasets[0].data_shape)
        self.batch_size = self.datasets[0].batch_size
        self.length = self.datasets[0].length
        self.labels = self.datasets[0].labels

    def batch(self):
        inputs = []
        outputs = None

        for dataset in self.datasets:
            batch_inputs, batch_outputs = dataset.batch()

            inputs.append(np.reshape(batch_inputs, [-1, 1, self.flattened_shape]))

            if outputs is None:
                outputs = batch_outputs

        return np.concatenate(inputs, axis=1), outputs

    def shuffle(self):
        indices = list(range(self.length))
        np.random.shuffle(indices)

        for dataset in self.datasets:
            dataset.shuffle(indices)
