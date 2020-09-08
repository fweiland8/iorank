import torch

from iorank.util.util import get_device

"""

Loss functions for the PyTorch context-sensitive ranking models.

"""


def hinged_rank_loss(y_pred, y_true):
    """
    Computes the hinged rank loss for the given data.
    
    :param y_pred: Predicted utility scores
    :param y_true: Ground truth ranking
    :return: Loss value
    """
    
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    y_true = y_true.to(get_device())
    y_pred = y_pred.to(get_device())
    greater_than_mask = torch.gt(y_true[:, None] - y_true[:, :, None], 0).float()

    minus = - torch.ones(y_true.size()[1], device=get_device())
    minus_mask = torch.eq(y_true, minus)
    minus_mask = torch.logical_not(minus_mask)
    minus_mask = minus_mask.float()
    minus_mask = minus_mask[:, None] * minus_mask[:, :, None]

    diff = 1 + y_pred[:, None] - y_pred[:, :, None]
    masked_diff = greater_than_mask * diff * minus_mask

    zeros = torch.zeros(masked_diff.size(), device=get_device())
    masked_diff = torch.max(masked_diff, zeros)
    loss = torch.sum(masked_diff, (1, 2)) / torch.max(torch.sum(greater_than_mask * minus_mask, (1, 2)),
                                                      torch.tensor(1.0, device=get_device()))
    return loss
