import random

import numpy as np
import torch
from typing import Optional


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def normalize_label(label, label_column, denormalize=False):

    label_stats = {"max": 30, "min": 6} 
    
    if not denormalize:
        return (label - label_stats["min"]) / (
            label_stats["max"] - label_stats["min"]
        )
    else:
        return label * (label_stats["max"] - label_stats["min"]) + label_stats["min"]



class AverageMeter(object):

    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    
    
def denormalize_relative(value):
    """
    Denormalize a relative value (only by the range)
    """
    label_stats = {"max": 30, "min": 6}

    return value * (label_stats["max"] - label_stats["min"])