import numpy as np
import os
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image
from torchvision import transforms
import h5py

class ChunkedGenerator:
    def __init__(self, batch_size, cameras, poses_3d, poses_2d, valid_frame,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all=False, train=True):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        pairs = []
        self.saved_index = {}
        start_index = 0

        if train == True:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_2d[key].shape[0] == poses_3d[key].shape[0]
                n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
                offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
                bounds = np.arange(n_chunks + 1) * chunk_length - offset
                augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                keys = np.tile(np.array(key).reshape([1, 3]), (len(bounds - 1), 1))
                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, reverse_augment_vector))
                if reverse_aug:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
                if augment:
                    if reverse_aug:
                        pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, ~reverse_augment_vector))
                    else:
                        pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))

                end_index = start_index + poses_3d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = start_index + poses_3d[key].shape[0]
        else:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_2d[key].shape[0] == poses_3d[key].shape[0]
                n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
                offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
                bounds = np.arange(n_chunks) * chunk_length - offset
                bounds_low = bounds[valid_frame[key].astype(bool)]
                bounds_high = bounds[valid_frame[key].astype(bool)] + np.ones(bounds_low.shape[0], dtype=int)

                augment_vector = np.full(len(bounds_low), False, dtype=bool)
                reverse_augment_vector = np.full(len(bounds_low), False, dtype=bool)
                keys = np.tile(np.array(key).reshape([1, 1]), (len(bounds_low), 1))
                pairs += list(zip(keys, bounds_low, bounds_high, augment_vector, reverse_augment_vector))
                if reverse_aug:
                    pairs += list(zip(keys, bounds_low, bounds_high, augment_vector, ~reverse_augment_vector))
                if augment:
                    if reverse_aug:
                        pairs += list(zip(keys, bounds_low, bounds_high, ~augment_vector, ~reverse_augment_vector))
                    else:
                        pairs += list(zip(keys, bounds_low, bounds_high, ~augment_vector, reverse_augment_vector))

                end_index = start_index + poses_3d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = start_index + poses_3d[key].shape[0]

        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-2], poses_2d[key].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

        self.valid_frame = valid_frame
        self.train = train

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        if self.train == True:
            subject, seq, cam_index = seq_i
            seq_name = (subject, seq, cam_index)
        else:
            seq_name = seq_i[0]
        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]

        if flip:
            self.batch_2d[:, :, 0] *= -1
            self.batch_2d[:, self.kps_left + self.kps_right] = self.batch_2d[:,
                                                               self.kps_right + self.kps_left]
        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                       ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]

            if flip:
                self.batch_3d[:, :, 0] *= -1
                self.batch_3d[:, self.joints_left + self.joints_right] = \
                    self.batch_3d[:, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[2] *= -1
                self.batch_cam[7] *= -1
        if self.train == True:
            if self.poses_3d is None and self.cameras is None:
                return None, None, self.batch_2d.copy(), seq, subject, int(cam_index)
            elif self.poses_3d is not None and self.cameras is None:
                return np.zeros(9), self.batch_3d.copy(), self.batch_2d.copy(), seq, subject, int(cam_index)
            elif self.poses_3d is None:
                return self.batch_cam, None, self.batch_2d.copy(), seq, subject, int(cam_index)
            else:
                return self.batch_cam, self.batch_3d.copy(), self.batch_2d.copy(), seq, subject, int(cam_index)
        else:
            return np.zeros(9), self.batch_3d.copy(), self.batch_2d.copy(), seq_name, None, None


class Fusion(Dataset):
    def __init__(self, args, train=True):
        """
        Dataset class for Fusion dataset.

        :param args: Argument parser or config object
        :param train: Boolean, whether to use training set or test set
        """
        self.train = train
        self.data_root = args.data.data_root
        self.n_frames = args.model.n_frames
        self.test_batch_size = args.data.test_batch_size
        self.test_aug = args.data.test_augmentation
        self.data_augmentation = args.data.data_augmentation
        self.reverse_augmentation = args.data.reverse_augmentation
        self.out_all = args.data.out_all

        pad = (self.n_frames - 1) // 2

        if self.train:
            self.poses_train, self.poses_train_2d = self.prepare_data(self.data_root, train=True)
            self.generator = ChunkedGenerator(
                self.test_batch_size, None, self.poses_train, self.poses_train_2d, None,
                chunk_length=1, pad=pad, augment=self.data_augmentation,
                reverse_aug=self.reverse_augmentation, kps_left=self.kps_left,
                kps_right=self.kps_right, joints_left=self.joints_left,
                joints_right=self.joints_right, out_all=self.out_all, train=True
            )
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.poses_test, self.poses_test_2d, self.valid_frame = self.prepare_data(self.data_root, train=False)
            self.generator = ChunkedGenerator(
                self.test_batch_size, None, self.poses_test, self.poses_test_2d, self.valid_frame,
                pad=pad, augment=False, kps_left=self.kps_left, kps_right=self.kps_right,
                joints_left=self.joints_left, joints_right=self.joints_right, train=False
            )

        self.key_index = self.generator.saved_index
        print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, path, train=True):
        """
        Prepares 2D and 3D pose data for training or testing.

        :param path: Path to the data directory
        :param train: Boolean, whether to use training or testing set
        :return: 2D and 3D pose data (and valid frames for testing)
        """
        out_poses_3d = {}
        out_poses_2d = {}
        valid_frame = {}

        self.kps_left, self.kps_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
        self.joints_left, self.joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]

        if train:
            data = np.load(os.path.join(path, "data_train_3dhp.npz"), allow_pickle=True)['data'].item()
            for seq in data.keys():
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    subject_name, seq_name = seq.split(" ")

                    data_3d = anim['data_3d']
                    data_3d[:, :14] -= data_3d[:, 14:15]
                    data_3d[:, 15:] -= data_3d[:, 14:15]
                    out_poses_3d[(subject_name, seq_name, cam)] = data_3d

                    data_2d = anim['data_2d']
                    data_2d[..., :2] = self.normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
                    confidence_scores = np.ones((*data_2d.shape[:2], 1))
                    data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
                    out_poses_2d[(subject_name, seq_name, cam)] = data_2d

            return out_poses_3d, out_poses_2d
        else:
            data = np.load(os.path.join(path, "data_test_3dhp.npz"), allow_pickle=True)['data'].item()
            for seq in data.keys():
                anim = data[seq]
                valid_frame[seq] = anim["valid"]

                data_3d = anim['data_3d']
                data_3d[:, :14] -= data_3d[:, 14:15]
                data_3d[:, 15:] -= data_3d[:, 14:15]
                out_poses_3d[seq] = data_3d

                data_2d = anim['data_2d']
                if seq in ["TS5", "TS6"]:
                    width, height = 1920, 1080
                else:
                    width, height = 2048, 2048

                data_2d[..., :2] = self.normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
                confidence_scores = np.ones((*data_2d.shape[:2], 1))
                data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
                out_poses_2d[seq] = data_2d

            return out_poses_3d, out_poses_2d, valid_frame

    @staticmethod
    def normalize_screen_coordinates(X, w, h):
        """
        Normalizes screen coordinates for 2D poses.

        :param X: 2D coordinates
        :param w: Frame width
        :param h: Frame height
        :return: Normalized coordinates
        """
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.

        :param index: Index of the sample
        :return: Data sample
        """
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
        cam, gt_3D, input_2D, seq, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)

        if not self.train and self.test_aug:
            _, _, input_2D_aug, _, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True,
                                                                   reverse=reverse)
            input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D
        scale = 1.0

        if self.train:
            return cam, gt_3D, input_2D_update, seq, scale, bb_box
        else:
            return cam, gt_3D, input_2D_update, seq, scale, bb_box
