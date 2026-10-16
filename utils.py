import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def normalize_label(label, label_column, denormalize=False):
    """
    Normalize or denormalize a label using min-max scaling.

    Parameters
    ----------
    label : torch.Tensor
        The raw or normalized label value.
    label_column : str
        The label type, e.g., 'grs' or 'grs_red'.
    denormalize : bool, optional
        If True, performs denormalization. Default is False (normalize).

    Returns
    -------
    torch.Tensor
        Normalized or denormalized label tensor.
    """
    label_stats = (
        {"max": 30, "min": 6} if label_column == "grs" else {"max": 15, "min": 3}
    )

    if not denormalize:
        return (label - label_stats["min"]) / (label_stats["max"] - label_stats["min"])
    else:
        return label * (label_stats["max"] - label_stats["min"]) + label_stats["min"]


class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, name: str, fmt: Optional[str] = ":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def denormalize_relative(value):
    """
    Denormalize a relative value (only by the range)
    """
    label_stats = {"max": 30, "min": 6}

    return value * (label_stats["max"] - label_stats["min"])
