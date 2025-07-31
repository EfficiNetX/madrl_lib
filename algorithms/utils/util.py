import numpy as np
import torch


def check(input):
    """
    入力が NumPy か PyTorch Tensor かわからない状況で、
    最終的に PyTorch Tensor として扱えるように統一するための関数
    """
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif isinstance(input, torch.Tensor):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch Tensor")
