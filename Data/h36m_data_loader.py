import os
import numpy as np
import torch
from torch.utils.data import Dataset
from Utils.Utils import read_pkl, flip_data


class Augmenter2D(object):
    """
        Make 2D augmentations on the fly. PyTorch batch-processing GPU version.
        Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/data/augmentation.py#L10
    """

    def __init__(self, args):
        self.d2c_params = read_pkl(args.d2c_params_path)
        self.noise = torch.load(args.noise_path)
        self.mask_ratio = args.mask_ratio
        self.mask_T_ratio = args.mask_T_ratio
        self.num_Kframes = 27
        self.noise_std = 0.002

    def dis2conf(self, dis, a, b, m, s):
        f = a / (dis + a) + b * dis
        shift = torch.randn(*dis.shape) * s + m
        # if torch.cuda.is_available():
        shift = shift.to(dis.device)
        return f + shift

    def add_noise(self, motion_2d):
        a, b, m, s = self.d2c_params["a"], self.d2c_params["b"], self.d2c_params["m"], self.d2c_params["s"]
        if "uniform_range" in self.noise.keys():
            uniform_range = self.noise["uniform_range"]
        else:
            uniform_range = 0.06
        motion_2d = motion_2d[:, :, :, :2]
        batch_size = motion_2d.shape[0]
        num_frames = motion_2d.shape[1]
        num_joints = motion_2d.shape[2]
        mean = self.noise['mean'].float()
        std = self.noise['std'].float()
        weight = self.noise['weight'][:, None].float()
        sel = torch.rand((batch_size, self.num_Kframes, num_joints, 1))
        gaussian_sample = (torch.randn(batch_size, self.num_Kframes, num_joints, 2) * std + mean)
        uniform_sample = (torch.rand((batch_size, self.num_Kframes, num_joints, 2)) - 0.5) * uniform_range
        noise_mean = 0
        delta_noise = torch.randn(num_frames, num_joints, 2) * self.noise_std + noise_mean
        # if torch.cuda.is_available():
        mean = mean.to(motion_2d.device)
        std = std.to(motion_2d.device)
        weight = weight.to(motion_2d.device)
        gaussian_sample = gaussian_sample.to(motion_2d.device)
        uniform_sample = uniform_sample.to(motion_2d.device)
        sel = sel.to(motion_2d.device)
        delta_noise = delta_noise.to(motion_2d.device)

        delta = gaussian_sample * (sel < weight) + uniform_sample * (sel >= weight)
        delta_expand = torch.nn.functional.interpolate(delta.unsqueeze(1), [num_frames, num_joints, 2],
                                                       mode='trilinear', align_corners=True)[:, 0]
        delta_final = delta_expand + delta_noise
        motion_2d = motion_2d + delta_final
        dx = delta_final[:, :, :, 0]
        dy = delta_final[:, :, :, 1]
        dis2 = dx * dx + dy * dy
        dis = torch.sqrt(dis2)
        conf = self.dis2conf(dis, a, b, m, s).clip(0, 1).reshape([batch_size, num_frames, num_joints, -1])
        return torch.cat((motion_2d, conf), dim=3)

    def add_mask(self, x):
        ''' motion_2d: (N,T,17,3)
        '''
        N, T, J, C = x.shape
        mask = torch.rand(N, T, J, 1, dtype=x.dtype, device=x.device) > self.mask_ratio
        mask_T = torch.rand(1, T, 1, 1, dtype=x.dtype, device=x.device) > self.mask_T_ratio
        x = x * mask * mask_T
        return x

    def augment2D(self, motion_2d, mask=False, noise=False):
        if noise:
            motion_2d = self.add_noise(motion_2d)
        if mask:
            motion_2d = self.add_mask(motion_2d)
        return motion_2d

class MotionDataset3D(Dataset):
    def __init__(self, args, subset_list, data_split, return_stats=False):
        """
        :param args: Arguments from the config file
        :param subset_list: A list of datasets
        :param data_split: Either 'train' or 'test'
        :param return_stats: Boolean to return additional stats if needed
        """
        np.random.seed(0)
        self.data_root = args.data.data_root
        self.add_velocity = args.data.add_velocity
        self.subset_list = subset_list
        self.data_split = data_split
        self.return_stats = return_stats
        self.flip = args.Augmentation.flip
        self.use_proj_as_2d = args.Augmentation.use_proj_as_2d

        self.file_list = self.generate_file_list()

    def generate_file_list(self):
        """
        Generates the file list from the subset directories.

        :return: List of file paths
        """
        file_list = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list.append(os.path.join(data_path, i))
        return file_list

    @staticmethod
    def construct_motion2d_by_projection(motion_3d):
        """
        Constructs 2D pose sequence by projecting the 3D pose orthographically.

        :param motion_3d: 3D motion data
        :return: 2D motion data
        """
        motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
        motion_2d[:, :, :2] = motion_3d[:, :, :2]  # Extract x and y from the 3D pose
        motion_2d[:, :, 2] = 1  # Set confidence score as 1
        return motion_2d

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        :param idx: Index of the sample
        :return: Tuple of 2D motion data, 3D motion data, and optional stats
        """
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        motion_2d = motion_file["data_input"]
        motion_3d = motion_file["data_label"]

        if motion_2d is None or self.use_proj_as_2d:
            motion_2d = self.construct_motion2d_by_projection(motion_3d)

        if self.add_velocity:
            motion_2d_coord = motion_2d[..., :2]
            velocity_motion_2d = motion_2d_coord[1:] - motion_2d_coord[:-1]
            motion_2d = motion_2d[:-1]
            motion_2d = np.concatenate((motion_2d, velocity_motion_2d), axis=-1)
            motion_3d = motion_3d[:-1]

        if self.data_split == 'train' and self.flip and np.random.rand() > 0.5:
            motion_2d = flip_data(motion_2d)
            motion_3d = flip_data(motion_3d)

        if self.return_stats:
            return (
                torch.FloatTensor(motion_2d),
                torch.FloatTensor(motion_3d),
                motion_file["mean"],
                motion_file["std"],
            )
        else:
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)
