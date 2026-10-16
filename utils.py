import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


def create_storage_folder(root, project, group, run_name):
    path = os.path.join(root, project, group, run_name)
    os.makedirs(path, exist_ok=True)
    return path


def set_torch_device(gpu_ids):
    if gpu_ids:
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
