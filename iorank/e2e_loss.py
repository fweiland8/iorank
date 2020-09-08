import torch

from iorank.util.util import get_device

"""
Loss functions for training the E2E model
"""


def e2e_loss(xywh_pred, confidence_pred, class_pred, scores_pred, xywh_gt, confidence_gt, classes_gt, rankings_gt, iou):
    """
    Applies the e2e loss to the given predictions.

    :param xywh_pred: Predicted bounding box coordinates
    :param confidence_pred: Predicted confidence scores
    :param class_pred: Predicted class probabilities
    :param scores_pred: Predicted utility scores
    :param xywh_gt: Ground truth bounding box coordinates
    :param confidence_gt: Ground truth confidence scores
    :param classes_gt: Ground truth class probabilities
    :param rankings_gt: Ground truth rankings
    :param iou: IoU values between predicted and ground truth boxes
    :return: Tuple of object detection loss and object ranking loss
    """

    r_loss = ranking_loss(scores_pred, rankings_gt, confidence_gt, iou)
    od_loss = yolo_loss(xywh_pred, confidence_pred, class_pred, xywh_gt, confidence_gt, classes_gt, iou)

    return od_loss, r_loss


def yolo_loss(xywh_pred, confidence_pred, class_pred, xywh_gt, confidence_gt, classes_gt, iou):
    """
    Computes the YOLO object detection loss to the provided ground truth and predicted data.

    :param xywh_pred: Predicted bounding box coordinates
    :param confidence_pred: Predicted confidence scores
    :param class_pred: Predicted class probabilities
    :param xywh_gt: Ground truth bounding box coordinates
    :param confidence_gt: Ground truth confidence scores
    :param classes_gt: Ground truth class probabilities
    :param iou: IoU values between predicted and ground truth boxes
    :return The loss value
    """

    contains_object_mask = confidence_gt.squeeze(-1)

    # Find responsible cells
    iou_max = torch.max(iou, dim=3, keepdim=True)[0]
    iou_max_mask = torch.eq(iou, iou_max).float()
    responsible = xywh_pred * iou_max_mask[:, :, :, :, None]
    responsible = torch.max(responsible, dim=3)[0]

    # Bounding box loss
    x_diff = responsible[:, :, :, 0] - xywh_gt[:, :, :, 0]
    x_diff_sq = torch.mul(x_diff, x_diff)

    y_diff = responsible[:, :, :, 1] - xywh_gt[:, :, :, 1]
    y_diff_sq = torch.mul(y_diff, y_diff)

    w_diff_rt = torch.sqrt(responsible[:, :, :, 2] + 1e-8) - torch.sqrt(xywh_gt[:, :, :, 2] + 1e-8)
    w_diff_rt_sq = torch.mul(w_diff_rt, w_diff_rt)

    h_diff_rt = torch.sqrt(responsible[:, :, :, 3] + 1e-8) - torch.sqrt(xywh_gt[:, :, :, 3] + 1e-8)
    h_diff_rt_sq = torch.mul(h_diff_rt, h_diff_rt)

    xy_loss = (x_diff_sq + y_diff_sq) * contains_object_mask
    xy_loss_sum = torch.sum(xy_loss, dim=(1, 2))

    wh_loss = (w_diff_rt_sq + h_diff_rt_sq) * contains_object_mask
    wh_loss_sum = torch.sum(wh_loss, dim=(1, 2))

    # Confidence loss (object)
    iou_contains_object_mask = contains_object_mask.unsqueeze(3) * iou_max_mask
    confidence_diff = confidence_pred - confidence_gt
    confidence_obj_loss = torch.mul(confidence_diff, confidence_diff) * iou_contains_object_mask
    confidence_obj_loss_sum = torch.sum(confidence_obj_loss, dim=(1, 2, 3))

    # Confidence loss (no object)
    iou_contains_object_mask_neg = (~(iou_contains_object_mask.bool())).float()
    confidence_noobj_loss = torch.mul(confidence_diff, confidence_diff) * iou_contains_object_mask_neg
    confidence_noobj_loss_sum = torch.sum(confidence_noobj_loss, dim=(1, 2, 3))

    # Class label loss
    cls_diff = class_pred - classes_gt
    cls_loss = torch.mul(cls_diff, cls_diff) * contains_object_mask.unsqueeze(3)
    cls_loss_sum = torch.sum(cls_loss, dim=(1, 2, 3))

    # Add individual losses up
    loss = 5 * xy_loss_sum + 5 * wh_loss_sum + confidence_obj_loss_sum + 0.5 * confidence_noobj_loss_sum + cls_loss_sum
    return loss


def ranking_loss(scores_pred, rankings_gt, confidence_gt, iou):
    """
    Compute the ranking loss for the given data.

    :param scores_pred: Predicted utility scores
    :param confidence_gt: Ground truth confidence scores
    :param rankings_gt: Ground truth rankings
    :param iou: IoU values between predicted and ground truth boxes
    :return The loss value
    """

    # Fetch predicted scores for the responsible cells (which have the higher IoU)
    iou_argmax = torch.argmax(iou, dim=3, keepdim=True)
    scores_pred_max = torch.gather(scores_pred, 3, iou_argmax)

    # Flatten tensors (remove grid structure)
    rankings_gt = rankings_gt.flatten(1)
    scores_pred_max = scores_pred_max.flatten(1)
    confidence_gt = confidence_gt.flatten(1)

    # Only consider cells where a ground truth object exists
    scores_pred_max = scores_pred_max * confidence_gt

    # Apply loss
    return e2e_hinged_rank_loss(scores_pred_max, rankings_gt, confidence_gt)


def e2e_hinged_rank_loss(y_pred, y_true, contains_object_mask):
    """
    Modified version of the hinged rank loss, which is adapted for training the E2E model.

    In contrast to the original hinged rank loss, the loss is not applied to all object but only
    to those objects for which there is a ground truth object (i.e. a '1' in the contains_object_mask)

    :param y_pred: Predicted utility scores
    :param y_true: Ground truth ranking
    :param contains_object_mask: Binary mask having a 1 in every position where a ground truth object exists
    :return: The loss value
    """

    greater_than_mask = torch.gt(y_true[:, None] - y_true[:, :, None], 0).float()
    contains_object_mask = contains_object_mask[:, None] * contains_object_mask[:, :, None]
    greater_than_mask = greater_than_mask * contains_object_mask

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
