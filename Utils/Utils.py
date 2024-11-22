import torch
import torch.optim as optim
import torch.nn as nn
import scipy.io as scio
import pickle
import os
import numpy as np
import copy
import json
import os
import random
import yaml
from easydict import EasyDict as edict
from torch.autograd import Variable
from typing import Any, IO


def denormalize(pred, seq):
    """
    Denormalizes predictions based on the resolution of each sequence.
    Args:
        pred (torch.Tensor): Normalized predictions tensor.
        seq (list): List of sequence names to determine resolution.
    Returns:
        torch.Tensor: Denormalized tensor with adjusted root-relative coordinates.
    """
    out = pred.detach().cpu().numpy()
    for idx in range(out.shape[0]):
        # Chọn độ phân giải phù hợp với mỗi sequence
        if seq[idx] in ['TS5', 'TS6']:
            res_w, res_h = 1920, 1080
        else:
            res_w, res_h = 2048, 2048
        out[idx, :, :, :2] = (out[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
        out[idx, :, :, 2:] = out[idx, :, :, 2:] * res_w / 2

    # Điều chỉnh lại gốc tọa độ của các joints với root joint
    out = out - out[..., 0:1, :]
    return torch.tensor(out, device=pred.device)

def decay_lr_exponentially(lr, lr_decay, optimizer):
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return lr

def get_variable(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json',):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def read_pkl(data_url):
    file = open(data_url, 'rb')
    content = pickle.load(file)
    file.close()
    return content

def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data


def resample(ori_len, target_len, replay=False, randomness=True):
    """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68"""
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len - target_len)
            return range(st, st + target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel * low + (1 - sel) * high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape) * interval + even
            result = np.clip(result, a_min=0, a_max=ori_len - 1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result


def split_clips(vid_list, n_frames, data_stride):
    """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L91"""
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i < len(vid_list):
        i += 1
        if i - st == n_frames:
            result.append(range(st, i))
            saved.add(vid_list[i - 1])
            st = st + data_stride
            n_clips += 1
        if i == len(vid_list):
            break
        if vid_list[i] != vid_list[i - 1]:
            if not (vid_list[i - 1] in saved):
                resampled = resample(i - st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i - 1])
            st = i
    return result

def save_checkpoint(model, optimizer, epoch, loss, best_mpjpe, checkpoint_type='latest', save_path='checkpoints'):
    """
    Save a checkpoint during training.

    Args:
        model: The model to save.
        optimizer: The optimizer used during training.
        epoch: Current epoch.
        loss: Current loss value.
        best_mpjpe: Best MPJPE so far.
        checkpoint_type: Either 'latest' or 'best'.
        save_path: Directory to save the checkpoint.
    """
    os.makedirs(save_path, exist_ok=True)
    filename = f"{checkpoint_type}_checkpoint.pth"
    checkpoint_path = os.path.join(save_path, filename)

    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_mpjpe': best_mpjpe,
        'lr': optimizer.param_groups[0]['lr']  # Save current learning rate
    }, checkpoint_path)

    print(
        f"{checkpoint_type.capitalize()} checkpoint saved at epoch {epoch} with loss {loss:.4f}. Path: {checkpoint_path}")

def save_data_inference(path, data_inference, latest):
    if latest:
        mat_path = os.path.join(path, 'inference_data.mat')
    else:
        mat_path = os.path.join(path, 'inference_data_best.mat')
    scio.savemat(mat_path, data_inference)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def print_args(args):
    print("[INFO] Input arguments:")
    for key, val in args.items():
        print(f"[INFO]   {key}: {val}")

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Define the skeleton pairs for training
def get_connections(num_joints, skeleton_pairs):
    connections = {i: [] for i in range(num_joints)}
    for joint1, joint2 in skeleton_pairs:
        connections[joint1].append(joint2)
        connections[joint2].append(joint1)
    return connections