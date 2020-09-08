import numpy as np
import torch

from iorank.util.util import is_dummy_box, pad, get_iou


def iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes

    :param box1: Bounding box coordinates in the form (x0,y0,x1,y1)
    :param box2: Bounding box coordinates in the form (x0,y0,x1,y1)
    :return: The IoU value of the two provided bounding boxes
    """

    # Compute intersection rectangle
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])

    intersection_area = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
    if intersection_area == 0:
        return 0.0

    # Compute areas of the individual boxes
    pred_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    gt_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    return intersection_area / (pred_area + gt_area - intersection_area)


def get_gt_to_pred_mappings(boxes_pred, boxes_gt):
    """
    Computes a mappings from ground truth objects to predicted objects.

    N: Batch size \n
    U: Upper bound for the number of objects (padding size) \n

    :param boxes_pred: Tensor with predicted bounding box coordinates. Size: (N,U,4)
    :param boxes_gt: Tensor with ground truth bounding box coordinates. Size: (N,U,4)
    :return: (pred_objects, gt_objects) Indices of objects in the predicted/ground truth data that correspond to each other.
    """

    threshold = 0.4
    all_pred_objects = []
    all_gt_objects = []

    n_instances = len(boxes_pred)
    # Iterate over all instances
    for i in range(n_instances):
        pred_objects = []
        gt_objects = []
        # Iterate over all ground truth boxes
        for j in range(len(boxes_gt[i])):
            # Ignore dummy boxes
            if is_dummy_box(boxes_gt[i][j]):
                continue
            accuracies = []
            # Compute IoU to each predicted box
            for k in range(len(boxes_pred[i])):
                if is_dummy_box(boxes_pred[i][k]):
                    accuracies.append(0)
                else:
                    accuracies.append(iou(boxes_gt[i][j], boxes_pred[i][k]))
            accuracies = np.array(accuracies)
            max_idx = np.argmax(accuracies)
            # Is there a predicted box exceeding the IoU threshold?
            if accuracies[max_idx] > threshold:
                gt_objects.append(j)
                pred_objects.append(max_idx)
        all_gt_objects.append(gt_objects)
        all_pred_objects.append(pred_objects)

    return all_pred_objects, all_gt_objects


def reduce_to_common_objects(pred, gt):
    """
    Reduces the predicted data and ground truth data to the 'actual' objects, i.e. the objects that occur in both
    ground truth data and predicted data.

    :param pred: Dict of predicted data (boxes, labels, confidence scores, ranking scores)
    :param gt: Dict of ground truth data (boxes, labels, confidence scores, ranking scores)
    :return: (pred,gt) Predicted and ground truth data, reduced to their common objects
    """

    all_pred_objects, all_gt_objects = get_gt_to_pred_mappings(pred["boxes"], gt["boxes"])

    n_objects = pred["boxes"].size()[1]
    n_instances = len(all_pred_objects)

    ret_pred = {key: [] for key in pred.keys()}
    ret_gt = {key: [] for key in gt.keys()}
    # Iterate over instances
    for i in range(n_instances):
        pred_objects = all_pred_objects[i]
        gt_objects = all_gt_objects[i]

        # Iterate over keys (boxes, labels, ..)
        for pred_key, gt_key in zip(pred.keys(), gt.keys()):
            # Apply padding to ensure unified size
            p = pred[pred_key][i][pred_objects]
            p = pad(p, n_objects)
            ret_pred[pred_key].append(p)

            g = gt[gt_key][i][gt_objects]
            g = pad(g, n_objects)
            ret_gt[gt_key].append(g)

    for pred_key, gt_key in zip(pred.keys(), gt.keys()):
        ret_pred[pred_key] = torch.stack(ret_pred[pred_key])
        ret_gt[gt_key] = torch.stack(ret_gt[gt_key])
    return ret_pred, ret_gt


def harmonic_mean(*args):
    """
    Computes the harmonic mean of multiple numbers.

    :type args: list
    :param args: An arbitrary number of numbers
    :return: The harmonic mean
    """

    n = len(args)
    sum = np.sum([1 / x for x in args])
    return n / sum


def get_object_detection_stats(boxes_pred, boxes_gt, conf_pred):
    """
    Computes object detection statistics for the provided ground truth boxes and predicted boxes.

    U: Upper bound for the number of objects (padding size)

    :param boxes_pred: Tensor of predicted bounding box coordinates of size (U,4)
    :param boxes_gt: Tensor of predicted bounding box coordinates of size (U,4)
    :param conf_pred: Tensor of predicted confidence scores of size U
    :return: (true_positives, n_pred, n_gt) The number of true
    positives, number of predicted objects and number of ground truth objects
    """

    threshold = 0.5
    # Ignore dummy boxes
    boxes_pred = boxes_pred[boxes_pred > -1].view(-1, 4)
    boxes_gt = boxes_gt[boxes_gt > -1].view(-1, 4)

    n_pred = boxes_pred.size(0)
    n_gt = boxes_gt.size(0)

    if n_pred == 0:
        return 0, n_pred, n_gt

    conf_sorted_idx = torch.argsort(conf_pred, descending=True)

    gt_found = torch.zeros(n_gt)

    for i in range(n_pred):
        for j in range(n_gt):
            idx = conf_sorted_idx[i]
            pred_box = boxes_pred[idx]
            gt_box = boxes_gt[j]

            # That GT box has already been found
            if gt_found[j] == 1:
                continue

            iou = get_iou(pred_box, gt_box)
            if iou > threshold:
                gt_found[j] = 1

    true_positives = int(torch.sum(gt_found))
    return true_positives, n_pred, n_gt
