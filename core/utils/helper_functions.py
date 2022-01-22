import numpy as np
import torch
import torch.nn.functional as F


def mean_squared_error_p(y_pred):
    """Modified mean square error that clips"""
    return torch.sum(torch.clamp((torch.max(y_pred ** 2, dim=-1)[0] - 10), 0.0, 100.0))


def exp_dec_error(y_pred, C=5):
    return torch.mean(
        torch.exp(
            -C * torch.sqrt(torch.clamp(torch.sum(y_pred ** 2, dim=-1), 0.000001, 10))
        )
    )


def cosine_proximity2(y_true, y_pred):
    """This loss is similar to the native cosine_proximity loss from Keras
    but it differs by the fact that only the two first components of the two vectors are used
    """

    y_true = F.normalize(y_true[:, 0:2], p=2, dim=-1)
    y_pred = F.normalize(y_pred[:, 0:2], p=2, dim=-1)
    return -torch.sum(y_true * y_pred, dim=-1)


def polar2euclid(r, t):
    return r * np.cos(t), r * np.sin(t)
