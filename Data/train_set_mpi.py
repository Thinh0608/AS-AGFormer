from torch.utils.data import Dataset
from Utils.Utils import flip_data
import torch
import numpy as np
import os
import random


class MPI3DHP(Dataset):
    def __init__(self, args, train=True):
        """
        Dataset class for MPI 3D Human Pose dataset.

        :param args: Argument parser or config object
        :param train: Boolean, whether to use training set or test set
        """
        self.train = train
        self.data_root = args.data.data_root
        self.flip = args.data.flip
        self.n_frames = args.model.n_frames
        self.stride = args.data.stride if train else args.model.n_frames
        self.poses_3d, self.poses_2d, self.poses_3d_valid_frames, self.seq_names = self.prepare_data()
        self.normalized_poses3d = self.normalize_poses()
        self.left_joints = [8, 9, 10, 2, 3, 4]
        self.right_joints = [11, 12, 13, 5, 6, 7]

    def normalize_poses(self):
        normalized_poses_3d = []
        if self.train:
            for pose_sequence in self.poses_3d:  # pose_sequence dim is (T, J, 3)
                normalized_sequence = self._normalize_pose_sequence(pose_sequence, 2048, 2048)
                normalized_poses_3d.append(normalized_sequence[None, ...])
        else:
            for seq_name, pose_sequence in zip(self.seq_names, self.poses_3d):  # pose_sequence dim is (T, J, 3)
                if seq_name in ["TS5", "TS6"]:
                    normalized_sequence = self._normalize_pose_sequence(pose_sequence, 1920, 1080)
                else:
                    normalized_sequence = self._normalize_pose_sequence(pose_sequence, 2048, 2048)
                normalized_poses_3d.append(normalized_sequence[None, ...])

        return np.concatenate(normalized_poses_3d, axis=0)

    def _normalize_pose_sequence(self, pose_sequence, width, height):
        """
        Normalize a single sequence of 3D poses.

        :param pose_sequence: ndarray of shape (T, J, 3)
        :param width: Frame width
        :param height: Frame height
        :return: Normalized pose sequence
        """
        normalized_sequence = pose_sequence.copy()
        normalized_sequence[..., :2] = normalized_sequence[..., :2] / width * 2 - [1, height / width]
        normalized_sequence[..., 2:] = normalized_sequence[..., 2:] / width * 2
        normalized_sequence = normalized_sequence - normalized_sequence[:, 14:15, :]
        return normalized_sequence

    def prepare_data(self):
        poses_2d, poses_3d, poses_3d_valid_frames, seq_names = [], [], [], []
        data_file = "data_train_3dhp.npz" if self.train else "data_test_3dhp.npz"
        data = np.load(os.path.join(self.data_root, data_file), allow_pickle=True)['data'].item()

        for seq in data.keys():
            if self.train:
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]
                    data_3d_partitioned, data_2d_partitioned, _ = self.extract_poses(anim, seq)
                    poses_3d.extend(data_3d_partitioned)
                    poses_2d.extend(data_2d_partitioned)
            else:
                anim = data[seq]
                valid_frames = anim['valid']
                data_3d_partitioned, data_2d_partitioned, valid_frames_partitioned = self.extract_poses(anim, seq,
                                                                                                        valid_frames)
                poses_3d.extend(data_3d_partitioned)
                poses_2d.extend(data_2d_partitioned)
                seq_names.extend([seq] * len(data_3d_partitioned))
                poses_3d_valid_frames.extend(valid_frames_partitioned)

        poses_3d = np.concatenate(poses_3d, axis=0)
        poses_2d = np.concatenate(poses_2d, axis=0)
        if len(poses_3d_valid_frames) > 0:
            poses_3d_valid_frames = np.concatenate(poses_3d_valid_frames, axis=0)

        return poses_3d, poses_2d, poses_3d_valid_frames, seq_names

    def extract_poses(self, anim, seq, valid_frames=None):
        """
        Extract and partition 3D and 2D poses.

        :param anim: Animation data
        :param seq: Sequence name
        :param valid_frames: Valid frames (for test set)
        :return: Partitioned 3D poses, 2D poses, and valid frames
        """
        data_3d = anim['data_3d']
        data_3d_partitioned, valid_frames_partitioned = self.partition(data_3d, valid_frames=valid_frames)

        data_2d = anim['data_2d']
        if seq in ["TS5", "TS6"]:
            width, height = 1920, 1080
        else:
            width, height = 2048, 2048

        data_2d[..., :2] = self.normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
        confidence_scores = np.ones((*data_2d.shape[:2], 1))
        data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
        data_2d_partitioned, _ = self.partition(data_2d)

        return data_3d_partitioned, data_2d_partitioned, valid_frames_partitioned

    @staticmethod
    def normalize_screen_coordinates(X, w, h):
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]

    def partition(self, data, clip_length=None, stride=None, valid_frames=None):
        clip_length = clip_length or self.n_frames
        stride = stride or self.stride
        data_list, valid_list = [], []
        n_frames = data.shape[0]

        for i in range(0, n_frames, stride):
            sequence = data[i:i + clip_length]
            if sequence.shape[0] == clip_length:
                data_list.append(sequence[None, ...])
            elif valid_frames is not None:
                new_indices = self.resample(sequence.shape[0], clip_length)
                extrapolated_sequence = sequence[new_indices]
                data_list.append(extrapolated_sequence[None, ...])

        return data_list, valid_list

    @staticmethod
    def resample(original_length, target_length):
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        result = np.floor(even)
        return np.clip(result, 0, original_length - 1).astype(np.uint32)

    def __getitem__(self, index):
        pose_2d = self.poses_2d[index]
        pose_3d_normalized = self.normalized_poses3d[index]

        if not self.train:
            valid_frames = self.poses_3d_valid_frames[index]
            pose_3d = self.poses_3d[index]
            seq_name = self.seq_names[index]
            return torch.FloatTensor(pose_2d), torch.FloatTensor(pose_3d_normalized), torch.FloatTensor(pose_3d), \
                   torch.IntTensor(valid_frames), seq_name

        if self.flip and random.random() > 0.5:
            pose_2d = flip_data(pose_2d, self.left_joints, self.right_joints)
            pose_3d_normalized = flip_data(pose_3d_normalized, self.left_joints, self.right_joints)

        return torch.FloatTensor(pose_2d), torch.FloatTensor(pose_3d_normalized)
