import logging
import numpy as np
import torch

from iorank.metrics.metric_utils import reduce_to_common_objects, get_object_detection_stats
from iorank.util.util import scores_to_rankings_torch, get_device, scores_to_rankings_with_size_torch

logger = logging.getLogger("metrics")


def label_accuracy(pred, gt):
    """
    Computes the label accuracy for the given predicted data and ground truth data.

    :param pred: Dict of predicted data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :param gt: Dict of ground truth data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :return: Average label accuracy for the provided instances
    """

    pred_labels = pred["labels"].to(get_device())
    gt_labels = gt["labels"].to(get_device())
    # Mask to distinguish between real and dummy data
    object_mask = torch.ne(gt_labels, -1).float()
    eq = torch.eq(pred_labels, gt_labels).float()
    eq = eq * object_mask

    sum_eq = torch.sum(eq, dim=1)
    n_objects = torch.sum(object_mask, dim=1)

    accuracies = sum_eq / (n_objects + 1e-8)
    return float(torch.mean(accuracies))


def average_ranking_size(pred, gt):
    """
    Computes the average ranking size, i.e. the average number of actual detected objects per image.

    :param pred: Dict of predicted data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :param gt: Dict of ground truth data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :return: Average ranking size
    """
    values = []
    for boxes_pred in pred["boxes"]:
        boxes_pred = boxes_pred[boxes_pred > -1].view(-1, 4)
        values.append(boxes_pred.size(0))
    return np.mean(values)


def object_detection_precision(pred, gt):
    """
    Computes the object detection precision for the given predicted and ground truth data.

    :param pred: Dict of predicted data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :param gt: Dict of ground truth data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :return: Average object detection precision for the provided instances
    """
    all_boxes_gt = gt["boxes"]
    all_boxes_pred = pred["boxes"]
    all_conf_pred = pred["conf"]

    values = []
    # Compute precision for each instance
    for boxes_pred, conf_pred, boxes_gt in zip(all_boxes_pred, all_conf_pred, all_boxes_gt):
        tp, n_pred, n_gt = get_object_detection_stats(boxes_pred, boxes_gt, conf_pred)
        precision = tp / (n_pred + 1e-8)

        values.append(precision)

    return np.mean(values)


def object_detection_recall(pred, gt):
    """
    Computes the object detection recall for the given predicted and ground truth data.

    :param pred: Dict of predicted data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :param gt: Dict of ground truth data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :return: Average object detection recall for the provided instances
    """
    all_boxes_gt = gt["boxes"]
    all_boxes_pred = pred["boxes"]
    all_conf_pred = pred["conf"]

    values = []
    # Compute recall for each instance
    for boxes_pred, conf_pred, boxes_gt in zip(all_boxes_pred, all_conf_pred, all_boxes_gt):
        tp, n_pred, n_gt = get_object_detection_stats(boxes_pred, boxes_gt, conf_pred)
        recall = tp / (n_gt + 1e-8)

        values.append(recall)

    return np.mean(values)


def kendalls_tau(pred, gt):
    """
    Computes the Kendall's Tau rank correlation coefficent for the given predicted and ground truth data.

    :param pred: Dict of predicted data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :param gt: Dict of ground truth data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :return: Average Kendall's Tau for the provided instances
    """
    # Metric is computed in rankings (not scores)
    rankings_gt = scores_to_rankings_torch(gt["scores"])
    # Ranking size = Number of objects in the image
    ranking_sizes = torch.sum(torch.ne(rankings_gt, -1), dim=1)
    rankings_pred = scores_to_rankings_with_size_torch(pred["scores"], ranking_sizes)

    result = []
    # Compute Kendall's Tau for each instance
    for ranking_pred, ranking_gt in zip(rankings_pred, rankings_gt):
        # Ignore dummy objects
        ranking_gt = ranking_gt[ranking_gt > -1]
        ranking_size = len(ranking_gt)
        ranking_pred = ranking_pred[:ranking_size]

        # Count discordant and concordant pairs
        discordant = 0
        concordant = 0
        # Trivial rankings
        if ranking_size == 0 or ranking_size == 1:
            result.append(1.0)
            continue
        for i in range(ranking_size):
            for j in range(i + 1, ranking_size):
                if ranking_gt[i] > ranking_gt[j] and ranking_pred[i] > ranking_pred[j]:
                    concordant += 1
                if ranking_gt[i] < ranking_gt[j] and ranking_pred[i] < ranking_pred[j]:
                    concordant += 1
                if ranking_gt[i] > ranking_gt[j] and ranking_pred[i] < ranking_pred[j]:
                    discordant += 1
                if ranking_gt[i] < ranking_gt[j] and ranking_pred[i] > ranking_pred[j]:
                    discordant += 1
        r = (concordant - discordant) / (concordant + discordant)
        result.append(r)
    return np.mean(result)


def spearman(pred, gt):
    """
    Computes the Spearman's Rho rank correlation coefficent for the given predicted and ground truth data.

    :param pred: Dict of predicted data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :param gt: Dict of ground truth data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :return: Average Spearman's Rho for the provided instances
    """
    # Metric is computed in rankings (not scores)
    rankings_gt = scores_to_rankings_torch(gt["scores"])
    ranking_sizes = torch.sum(torch.ne(rankings_gt, -1), dim=1)
    rankings_pred = scores_to_rankings_with_size_torch(pred["scores"], ranking_sizes)

    # Do metric computation on CUDA
    rankings_pred = rankings_pred.to(get_device())
    rankings_gt = rankings_gt.to(get_device())

    # Object mask has entry 1 where there is a real object (no dummy object)
    object_mask = rankings_gt.ne(-1).float()
    n_objects = torch.sum(object_mask, dim=1)

    diff = rankings_pred - rankings_gt
    diff = diff * object_mask
    diff_sq = diff ** 2
    sums = torch.sum(diff_sq, 1)
    result = 1 - (6.0 * sums) / torch.max((n_objects * (n_objects - 1) * (n_objects + 1)),
                                          torch.tensor(1.0, device=get_device()))
    # Move result back to CPU
    result = result.cpu().numpy()
    return np.mean(result)


def zero_one_accuracy(pred, gt):
    """
    Computes the 0/1 ranking accuracy for the given predicted and ground truth data.

    :param pred: Dict of predicted data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :param gt: Dict of ground truth data (boxes, labels, confidence scores, ranking scores) for a batch of data
    :return: Average 0/1 ranking accuracy for the provided instances
    """
    # Metric is computed in rankings (not scores)
    rankings_gt = scores_to_rankings_torch(gt["scores"])
    ranking_sizes = torch.sum(torch.ne(rankings_gt, -1), dim=1)
    rankings_pred = scores_to_rankings_with_size_torch(pred["scores"], ranking_sizes)

    # Do computation on CUDA
    rankings_pred = rankings_pred.to(get_device())
    rankings_gt = rankings_gt.to(get_device())

    result = torch.eq(rankings_pred, rankings_gt).all(1).float()
    return np.mean(result.cpu().numpy())


def reduced_ranking(metric):
    """
    Wraps the given metric such that the ground truth data and predicted data are reduced to their common objects
    before the metric is computed. This is required for label and ranking metrics.

    :param metric: Metric to be wrapped
    :return: Wrapped metric
    """

    def m(pred, gt):
        # Reduce to common objects before applying the metric
        pred, gt = reduce_to_common_objects(pred, gt)
        return metric(pred, gt)

    return m
