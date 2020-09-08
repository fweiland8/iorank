import logging
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from iorank.e2e_loss import e2e_loss
from iorank.objectdetection.backbone import MobileNetBackbone, ResNetBackbone
from iorank.objectdetection.yolo import YOLODetectionModule
from iorank.util.util import get_device, pad, get_iou, get_spatial_mask
from torchcsrank.core.fate_network import FATENetwork


class ImageObjectRankerE2E(nn.Module):
    def __init__(self, n_classes, max_n_objects, n_cells=7, predictions_per_cell=2, input_size=448,
                 backbone_name='mobilenet', conv_feature_size=3, confidence_threshold=0.1, nms_threshold=0.7,
                 include_spatial_mask=False, feature_reduction=None):
        """
        Creates an instance of the E2E image object ranker.

        :param n_classes: Number of classes in the dataset
        :param max_n_objects: Maximum number of objects to be considered and padding size
        :param n_cells: Number of grid cells (for the YOLO object detector)
        :param predictions_per_cell: Number of predictions per grid cell (for the YOLO object detector)
        :param input_size: The size to which input images are (down)scaled
        :param backbone_name: Name of the backbone network to be used, either 'mobilenet' or 'resnet'
        :param conv_feature_size: Side length of the feature map generated for detected objects
        :param confidence_threshold: Confidence a detection must have in order to be retained
        :param nms_threshold: Threshold value for Non Maximum Suppression
        :param include_spatial_mask: Enables appending a spatial mask to the generated feature vectors
        :param feature_reduction: Enables reducing the dimensionality of the convolutional feature map provided by the backbone. Value out of ('mean','conv') must be provided.
        """
        super(ImageObjectRankerE2E, self).__init__()
        self.logger = logging.getLogger(ImageObjectRankerE2E.__name__)

        self.backbone_name = backbone_name
        self.input_size = input_size
        self.n_classes = n_classes
        self.max_n_objects = max_n_objects
        self.n_cells = n_cells
        self.predictions_per_cell = predictions_per_cell
        self.conv_feature_size = conv_feature_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.include_spatial_mask = include_spatial_mask
        self.mask_size = 64

        self.feature_reduction = feature_reduction
        if self.feature_reduction is not None and self.feature_reduction != 'mean' and self.feature_reduction != 'conv':
            raise RuntimeError("Invalid feature_reduction value: {}".format(self.feature_reduction))

        # Number of features produced by the object detector
        out_features = self.n_cells * self.n_cells * (self.predictions_per_cell * 5 + self.n_classes)

        if self.backbone_name == 'mobilenet':
            self.backbone = MobileNetBackbone(self.input_size)
        elif self.backbone_name == 'resnet':
            self.backbone = ResNetBackbone(self.input_size)
        else:
            raise RuntimeError("Invalid backbone name : {}".format(self.backbone_name))

        # Construct YOLO classifier dependent on the convolutional feature map size provided by the backbone
        conv_map_size = int(self.backbone.conv_map_size)
        conv_map_depth = int(self.backbone.conv_map_depth)
        self.classifier = YOLODetectionModule(conv_map_size=conv_map_size,
                                              conv_map_depth=conv_map_depth,
                                              feature_reduction=self.feature_reduction,
                                              predictions_per_cell=self.predictions_per_cell,
                                              out_features=out_features,
                                              n_cells=self.n_cells,
                                              n_classes=self.n_classes)
        # Reduction factor in the backbone
        self.spatial_scale = 1.0 / 32.0
        self.cell_size = int(self.input_size / self.n_cells)

        n_object_features = self.conv_feature_size ** 2 * conv_map_depth

        if self.include_spatial_mask:
            n_object_features += self.mask_size ** 2
        self.ranker = FATENetwork(n_object_features)

    def forward(self, x, targets=None):
        """
        Processes the given input images. While training, targets have to be passed.
        
        N: Batch size
        
        :param x: Tensor of input images of size (N,3,H,W)
        :param targets: Ground truth data. Required only in training mode. 
        :return: In training mode, a loss value, in evaluation mode the predictions
        """

        boxes = None
        if self.training:
            if targets is None:
                raise RuntimeError("Model is in training mode but no targets are provided!")
            else:
                boxes = targets["boxes"]

        # Preparations
        batch_size = x.size(0)
        input_h = x.size()[-2]
        input_w = x.size()[-1]
        height_factor, width_factor = self._get_scaling_factors(input_h, input_w)
        x, boxes = self._resize_down(x, height_factor, width_factor, boxes=boxes)

        # 1. Backbone
        conv_feature_map = self.backbone(x)

        # 2. YOLO object detector
        xywh, confidence, cls_scores = self.classifier(conv_feature_map)
        boxes = self._xywh_to_boxes(xywh, input_h, input_w)

        if self.training:

            # Generate feature vectors for each detected object
            all_features = []
            for i in range(batch_size):
                image_boxes = boxes[i]
                # 3. RoIAlign
                conv_box_features = self._get_conv_features(i, image_boxes, conv_feature_map)

                # Optionally append spatial masks to the feature vectors
                if self.include_spatial_mask:
                    spatial_masks = [get_spatial_mask(box, input_h, input_w, mask_size=self.mask_size) for box in
                                     image_boxes]
                    spatial_masks = torch.stack(spatial_masks)
                    features = torch.cat((conv_box_features, spatial_masks), dim=1)
                else:
                    features = conv_box_features

                all_features.append(features)
            all_features = torch.stack(all_features)

            # 4. Rank all objects while training
            ranking_scores = self.ranker(all_features)
            ranking_scores = ranking_scores.view(-1, self.n_cells, self.n_cells, self.predictions_per_cell)

            # 5. Compute loss
            xywh_gt, confidence_gt, classes_gt, scores_gt = self._prepare_gt(targets)
            ious = self._get_ious(xywh, xywh_gt)
            losses = e2e_loss(xywh, confidence, cls_scores, ranking_scores, xywh_gt, confidence_gt, classes_gt,
                              scores_gt,
                              ious)

            return losses
        else:
            # 3. RoIAlign
            conv_features = []
            for i in range(batch_size):
                topk_boxes = boxes[i]
                conv_box_features = self._get_conv_features(i, topk_boxes, conv_feature_map)
                conv_features.append(conv_box_features)
            conv_features = torch.stack(conv_features)

            boxes_pred, conf_pred, cls_scores_pred = self._postprocess(xywh, confidence, cls_scores)
            labels = torch.argmax(cls_scores_pred, dim=2)

            all_features = []
            all_boxes = []
            all_labels = []
            all_confs = []
            for i in range(batch_size):
                b = boxes_pred[i]
                l = labels[i]
                conf = conf_pred[i]
                features = conv_features[i]

                # Filter out detections with a too low confidence
                above_threshold_idx = torch.where(conf > self.confidence_threshold)
                b = b[above_threshold_idx]
                conf = conf[above_threshold_idx]
                l = l[above_threshold_idx]
                features = features[above_threshold_idx]

                # Optionally append spatial masks to the feature vectors
                if self.include_spatial_mask:
                    if b.size(0) > 0:
                        spatial_masks = [get_spatial_mask(box, input_h, input_w, mask_size=self.mask_size) for box in b]
                        spatial_masks = torch.stack(spatial_masks)
                        features = torch.cat((features, spatial_masks), dim=1)
                    else:
                        # Resize empty tensor to make stack work later
                        features = features.view(0, features.size(1) + self.mask_size ** 2)

                # 4. NMS
                if b.size(0) > 0:
                    b, conf, l, features = self._nms(b, conf, l, features)

                # 5. Take top k detections if there are too many detections
                objects_found = b.size(0)
                if objects_found > self.max_n_objects:
                    topk_idx = torch.topk(conf, k=self.max_n_objects)[1]
                    b = b[topk_idx]
                    l = l[topk_idx]
                    features = features[topk_idx]
                    conf = conf[topk_idx]

                b = self._resize_up(b, height_factor, width_factor)

                # Padding
                b = pad(b, self.max_n_objects)
                l = pad(l, self.max_n_objects)
                features = pad(features, self.max_n_objects)
                conf = pad(conf, self.max_n_objects)

                all_boxes.append(b)
                all_labels.append(l)
                all_features.append(features)
                all_confs.append(conf)

            all_boxes = torch.stack(all_boxes)
            all_labels = torch.stack(all_labels)
            all_features = torch.stack(all_features)
            all_confs = torch.stack(all_confs)

            # 6. Object ranking
            ranking_scores = self.ranker(all_features)

            ret = {"boxes": all_boxes, "labels": all_labels, "scores": ranking_scores, "conf": all_confs}

            return ret

    def _nms(self, boxes, confidences, cls, features):
        """
        Applies Non Maximum Suppression (NMS).
        
        :param boxes: Bounding box coordinates
        :param confidences: Confidence scores
        :param cls: Class labels
        :param features: Feature vectors
        :return: Predictions, to which NMS has been applied
        """

        # Sort boxes according to confidence
        sorted_idx = torch.argsort(confidences, descending=True)
        idx_list = sorted_idx.tolist()

        ret_idx = []

        while len(idx_list) > 0:
            # Get prediction with current maximum confidence
            max_idx = idx_list[0]
            max_box = boxes[max_idx]
            ret_idx.append(max_idx)
            del idx_list[0]

            to_delete = []
            # Is there another similar detection? If yes, delete it
            for i in range(len(idx_list)):
                idx = idx_list[i]
                box = boxes[idx]
                iou = get_iou(max_box, box)
                if iou > self.nms_threshold:
                    to_delete.append(i)

            idx_list = [idx_list[i] for i in range(len(idx_list)) if i not in to_delete]

        boxes = boxes[ret_idx]
        confidences = confidences[ret_idx]
        cls = cls[ret_idx]
        features = features[ret_idx]
        return boxes, confidences, cls, features

    def _get_scaling_factors(self, h, w):
        """
        Computes the factors by which the input has to be scaled in order to get the desired input size.

        :param h: Image height
        :param w: Image width
        :return: Tuple of scaling factors for height and width
        """

        height_factor = float(self.input_size) / h
        width_factor = float(self.input_size) / w
        return height_factor, width_factor

    def _resize_down(self, images, height_factor, width_factor, boxes=None):
        """
        Resizes the given images down to the desired input size using the given scaling factors. Optionally, also
        bounding boxes can be scaled down using the same scaling factors.

        :param images: Images to be scaled down
        :param height_factor: Scaling factor for the height
        :param width_factor: Scaling factor for the width
        :param boxes: Optional. Bounding box coordinates
        :return: Resized images and optionally, resized bounding boxes
        """

        resized_images = torch.nn.functional.interpolate(images, scale_factor=(height_factor, width_factor),
                                                         mode='bilinear')
        resized_boxes = None
        if boxes is not None:
            resized_boxes = boxes
            resized_boxes[:, :, [0, 2]] *= width_factor
            resized_boxes[:, :, [1, 3]] *= height_factor
        return resized_images, resized_boxes

    def _resize_up(self, boxes, height_factor, width_factor):
        """
        Resizes the given bounding boxes up the given scaling factors.

        :param boxes: Bounding box coordinates to be resized up
        :param height_factor: Scaling factor for the height
        :param width_factor: Scaling factor for the width
        :return: Resized bounding boxes
        """

        resized_boxes = boxes
        resized_boxes[:, [0, 2]] = resized_boxes[:, [0, 2]] / width_factor
        resized_boxes[:, [1, 3]] = resized_boxes[:, [1, 3]] / height_factor
        return resized_boxes

    def _xywh_to_boxes(self, xywh, h, w):
        """
        Converts the bounding boxes in the (x,y,w,h) format to the (x0,y0,x1,y1) format.
        
        :param xywh: Bounding boxes in the (x,y,w,h) format
        :param h: Image height
        :param w: Image width
        :return: Bounding boxes in the (x0,y0,x1,y1) format
        """

        batch_size = xywh.size(0)
        xywh = xywh.clone()

        # x,y are given relative to the cell
        xywh[:, :, :, :, [0, 1]] = xywh[:, :, :, :, [0, 1]] * self.cell_size
        for i in range(self.n_cells):
            for j in range(self.n_cells):
                xywh[:, i, j, :, 0] = xywh[:, i, j, :, 0] + i * self.cell_size
                xywh[:, i, j, :, 1] = xywh[:, i, j, :, 1] + j * self.cell_size

        # w,h are given relative to the image size
        xywh[:, :, :, :, [2, 3]] = xywh[:, :, :, :, [2, 3]] * self.input_size

        xywh = xywh.view(batch_size, -1, 4)
        boxes = torch.empty(xywh.size(), device=get_device())

        boxes[:, :, 0] = xywh[:, :, 0] - 0.5 * xywh[:, :, 2]
        boxes[:, :, 1] = xywh[:, :, 1] - 0.5 * xywh[:, :, 3]
        boxes[:, :, 2] = xywh[:, :, 0] + 0.5 * xywh[:, :, 2]
        boxes[:, :, 3] = xywh[:, :, 1] + 0.5 * xywh[:, :, 3]

        # Correct values outside the image
        boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]], min=0, max=w)
        boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]], min=0, max=h)

        return boxes

    def _get_conv_features(self, batch_no, boxes, conv_feature_map):
        """
        Extract convolutional features from the given feature map for the provided boxes using RoIAlign

        :param batch_no: Number of the considered instance in the batch
        :param boxes: Bounding box coordinates
        :param conv_feature_map: Convolutional feature map from which the features are to be extracted
        :return: Convolutional features for the boxes
        """
        # In order to make RoIAlign work, boxes must be enriched with the batch_no
        boxes = F.pad(boxes, (1, 0), mode='constant', value=batch_no)
        features = roi_align(conv_feature_map, boxes, (self.conv_feature_size, self.conv_feature_size),
                             spatial_scale=self.spatial_scale, sampling_ratio=2)
        features = features.flatten(1)
        return features

    def _postprocess(self, xywh_, confidence, cls_scores):
        """
        Postprocesses the provided predictions of the YOLO classifier.

        :param xywh_: Predited bounding boxes in the format (x,y,w,h)
        :param confidence: Predicted confidence scores
        :param cls_scores: Predicted class probabilites
        :return: Postprocessed data
        """
        batch_size = xywh_.size(0)

        xywh = xywh_.clone()

        # x,y are given relative to the cell
        xywh[:, :, :, :, 0] = xywh[:, :, :, :, 0] * self.cell_size
        xywh[:, :, :, :, 1] = xywh[:, :, :, :, 1] * self.cell_size

        for i in range(batch_size):
            for j in range(self.n_cells):
                for k in range(self.n_cells):
                    xywh[i, j, k, :, 0] = xywh[i, j, k, :, 0] + j * self.cell_size
                    xywh[i, j, k, :, 1] = xywh[i, j, k, :, 1] + k * self.cell_size

        # w,h are given relative to the image size
        xywh[:, :, :, :, 2] = xywh[:, :, :, :, 2] * self.input_size
        xywh[:, :, :, :, 3] = xywh[:, :, :, :, 3] * self.input_size

        # Turn xywh boxes into (x0,y0,x1,y1) boxes
        boxes = torch.empty(batch_size, self.n_cells, self.n_cells, self.predictions_per_cell, 4)
        boxes[:, :, :, :, 0] = xywh[:, :, :, :, 0] - 0.5 * xywh[:, :, :, :, 2]
        boxes[:, :, :, :, 1] = xywh[:, :, :, :, 1] - 0.5 * xywh[:, :, :, :, 3]
        boxes[:, :, :, :, 2] = xywh[:, :, :, :, 0] + 0.5 * xywh[:, :, :, :, 2]
        boxes[:, :, :, :, 3] = xywh[:, :, :, :, 1] + 0.5 * xywh[:, :, :, :, 3]

        boxes[:, :, :, :, 0] = torch.clamp(boxes[:, :, :, :, 0], min=0, max=1241)
        boxes[:, :, :, :, 1] = torch.clamp(boxes[:, :, :, :, 1], min=0, max=374)
        boxes[:, :, :, :, 2] = torch.clamp(boxes[:, :, :, :, 2], min=0, max=1241)
        boxes[:, :, :, :, 3] = torch.clamp(boxes[:, :, :, :, 3], min=0, max=374)

        # Class scores = Class probability * Confidence for that cell
        cls = cls_scores.view(batch_size, self.n_cells ** 2, self.n_classes)
        c = confidence.view(batch_size, self.n_cells ** 2, self.predictions_per_cell, 1)
        cls_return = torch.empty(batch_size, self.n_cells ** 2, self.predictions_per_cell, self.n_classes)
        for i in range(c.size(0)):
            for j in range(c.size(1)):
                cls_return[i][j] = cls[i][j] * c[i][j]

        # Remove grid organization of the data
        boxes = boxes.view(batch_size, -1, 4)
        cls_return = cls_return.view(batch_size, self.n_cells ** 2 * self.predictions_per_cell, self.n_classes)
        c = c.view(batch_size, self.n_cells ** 2 * self.predictions_per_cell)
        return boxes, c, cls_return

    def _prepare_gt(self, gt):
        """
        Prepares the given ground truth data, such that the E2E loss can easily applied.

        Especially, the data are organized in a grid structure.

        :param gt: Ground truth data (boxes,labels,scores)
        :return: Prepared data for training
        """
        boxes_gt = gt["boxes"]
        labels_gt = gt["labels"]
        ranking_scores_gt = gt["scores"]
        batch_size = boxes_gt.size(0)

        # Create empty tensors
        xywh = torch.zeros(batch_size, self.n_cells, self.n_cells, 4, device=get_device())
        classes = torch.zeros(batch_size, self.n_cells, self.n_cells, self.n_classes, device=get_device())
        confidence = torch.zeros(batch_size, self.n_cells, self.n_cells, 1, device=get_device())
        # Initialize ranking scores with -1
        scores = -1 * torch.ones(batch_size, self.n_cells, self.n_cells, 1, device=get_device())

        for image_no in range(batch_size):
            image_boxes = boxes_gt[image_no]
            image_labels = labels_gt[image_no]
            image_scores = ranking_scores_gt[image_no]

            for object_no, (box, label, score) in enumerate(zip(image_boxes, image_labels, image_scores)):
                # Ignore dummy objects
                if torch.max(box) == -1 or label == -1 or score == -1:
                    continue
                # Ignore boxes that are outside of the image
                if box[0] >= self.input_size and box[2] >= self.input_size or box[1] >= self.input_size and \
                        box[3] >= self.input_size:
                    continue

                # Turn (x0,y0,x1,y1) box to (x,y,w,h) box
                w = box[2] - box[0]
                h = box[3] - box[1]
                x = box[0] + 0.5 * w
                y = box[1] + 0.5 * h

                # Find responsible cell
                cell_x = int(x // self.cell_size)
                cell_y = int(y // self.cell_size)

                # Handle boxes at the image border
                if cell_x == self.n_cells:
                    cell_x -= 1
                if cell_y == self.n_cells:
                    cell_y -= 1

                xywh[image_no, cell_x, cell_y][0] = (x % self.cell_size) / self.cell_size
                xywh[image_no, cell_x, cell_y][1] = (y % self.cell_size) / self.cell_size
                xywh[image_no, cell_x, cell_y][2] = w / float(self.input_size)
                xywh[image_no, cell_x, cell_y][3] = h / float(self.input_size)
                # Confidence is always 1 if there is a box inside that cell
                confidence[image_no, cell_x, cell_y] = 1.0

                cls_label = int(label)
                classes[image_no, cell_x, cell_y, cls_label] = 1.0

                scores[image_no, cell_x, cell_y] = score

            # Convert scores to ranking
            n_objects = torch.sum(confidence[image_no])

            tmp = scores[image_no]
            tmp = tmp.flatten(0)
            tmp = torch.argsort(tmp)
            tmp = torch.flip(tmp, dims=[0])
            tmp = torch.argsort(tmp)
            tmp[tmp > n_objects - 1] = 0
            tmp = tmp.view(self.n_cells, self.n_cells, 1)
            scores[image_no] = tmp

        return xywh, confidence, classes, scores

    def _get_ious(self, _xywh_pred, _xywh_gt):
        """
        Computes IoU values for the given predicted boxes and ground truth boxes.

        :param _xywh_pred: Predicted bounding box coordinates in the form (x,y,w,h)
        :param _xywh_gt: Ground truth bounding box coordinates in the form (x,y,w,h)
        :return: A tensor of IoU values
        """
        xywh_pred = _xywh_pred.clone()
        xywh_gt = _xywh_gt.clone()

        # Scale to pixels
        xywh_pred[:, :, :, :, 0] = xywh_pred[:, :, :, :, 0] * self.cell_size
        xywh_pred[:, :, :, :, 1] = xywh_pred[:, :, :, :, 1] * self.cell_size
        xywh_pred[:, :, :, :, 2] = xywh_pred[:, :, :, :, 2] * self.input_size
        xywh_pred[:, :, :, :, 3] = xywh_pred[:, :, :, :, 3] * self.input_size
        xywh_gt[:, :, :, 0] = xywh_gt[:, :, :, 0] * self.cell_size
        xywh_gt[:, :, :, 1] = xywh_gt[:, :, :, 1] * self.cell_size
        xywh_gt[:, :, :, 2] = xywh_gt[:, :, :, 2] * self.input_size
        xywh_gt[:, :, :, 3] = xywh_gt[:, :, :, 3] * self.input_size

        for i in range(self.n_cells):
            for j in range(self.n_cells):
                xywh_pred[:, i, j, :, 0] = xywh_pred[:, i, j, :, 0] + i * self.cell_size
                xywh_pred[:, i, j, :, 1] = xywh_pred[:, i, j, :, 1] + j * self.cell_size

                xywh_gt[:, i, j, 0] = xywh_gt[:, i, j, 0] + i * self.cell_size
                xywh_gt[:, i, j, 1] = xywh_gt[:, i, j, 1] + j * self.cell_size

        # Turn (x,y,w,h) to (x0,y0,x1,y1)
        bbox_x0_pred = xywh_pred[:, :, :, :, 0] - 0.5 * xywh_pred[:, :, :, :, 2]
        bbox_y0_pred = xywh_pred[:, :, :, :, 1] - 0.5 * xywh_pred[:, :, :, :, 3]
        bbox_x1_pred = xywh_pred[:, :, :, :, 0] + 0.5 * xywh_pred[:, :, :, :, 2]
        bbox_y1_pred = xywh_pred[:, :, :, :, 1] + 0.5 * xywh_pred[:, :, :, :, 3]

        bbox_x0_gt = xywh_gt[:, :, :, None, 0] - 0.5 * xywh_gt[:, :, :, None, 2]
        bbox_y0_gt = xywh_gt[:, :, :, None, 1] - 0.5 * xywh_gt[:, :, :, None, 3]
        bbox_x1_gt = xywh_gt[:, :, :, None, 0] + 0.5 * xywh_gt[:, :, :, None, 2]
        bbox_y1_gt = xywh_gt[:, :, :, None, 1] + 0.5 * xywh_gt[:, :, :, None, 3]

        # Actual IoU computation
        x0 = torch.max(bbox_x0_pred, bbox_x0_gt)
        y0 = torch.max(bbox_y0_pred, bbox_y0_gt)
        x1 = torch.min(bbox_x1_pred, bbox_x1_gt)
        y1 = torch.min(bbox_y1_pred, bbox_y1_gt)

        intersection_width = torch.max(x1 - x0 + 1, torch.tensor([0.0], device=get_device()))
        intersection_height = torch.max(y1 - y0 + 1, torch.tensor([0.0], device=get_device()))

        intersection_area = intersection_width * intersection_height
        union_area = xywh_pred[:, :, :, :, 2] * xywh_pred[:, :, :, :, 3] + xywh_gt[:, :, :, None, 2] * xywh_gt[:, :, :,
                                                                                                       None, 3]
        iou = intersection_area / (union_area - intersection_area)
        return iou
