import logging
import torch
from torch import nn

from iorank.e2e_loss import yolo_loss
from iorank.objectdetection.backbone import MobileNetBackbone, ResNetBackbone
from iorank.training.yolo_trainer import YOLOTrainer
from iorank.util.util import get_device, pad, get_iou


class YOLODetectionModule(nn.Module):
    def __init__(self, conv_map_size, conv_map_depth, feature_reduction, out_features, n_cells, predictions_per_cell,
                 n_classes):
        super(YOLODetectionModule, self).__init__()
        self.n_cells = n_cells
        self.predictions_per_cell = predictions_per_cell
        self.n_classes = n_classes
        self.conv_map_size = conv_map_size
        self.conv_map_depth = conv_map_depth
        self.feature_reduction = feature_reduction

        if self.feature_reduction == 'conv':
            self.conv = nn.Conv2d(conv_map_depth, 128, kernel_size=(1, 1), stride=(2, 2))
            n_conf_features = int((conv_map_size / 2) ** 2 * 128)
        elif self.feature_reduction == 'mean':
            self.conv = nn.Conv2d(conv_map_depth, conv_map_depth, kernel_size=(1, 1), stride=(2, 2))
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            n_conf_features = conv_map_depth
        elif self.feature_reduction is None:
            self.conv = nn.Conv2d(conv_map_depth, conv_map_depth, kernel_size=(1, 1), stride=(2, 2))
            n_conf_features = int((conv_map_size / 2) ** 2 * conv_map_depth)
        self.fc1 = nn.Linear(n_conf_features, 4096)
        self.fc2 = nn.Linear(4096, out_features)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.conv(x)

        if self.feature_reduction == 'mean':
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = torch.dropout(x, 0.5, self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        xywh, confidence_scores, cls_pred = self._postprocess_result(x)
        return xywh, confidence_scores, cls_pred

    def _postprocess_result(self, x):
        batch_size = x.size(0)
        xywh = x[:, :self.n_cells ** 2 * self.predictions_per_cell * 4]
        confidence_scores = x[:,
                            self.n_cells ** 2 * self.predictions_per_cell * 4:self.n_cells ** 2 * self.predictions_per_cell * 5]
        cls_pred = x[:, self.n_cells ** 2 * self.predictions_per_cell * 5:]

        xywh = xywh.view(batch_size, self.n_cells, self.n_cells, self.predictions_per_cell, 4)
        confidence_scores = confidence_scores.view(batch_size, self.n_cells, self.n_cells, self.predictions_per_cell)
        cls_pred = cls_pred.view(batch_size, self.n_cells, self.n_cells, self.n_classes)
        return xywh, confidence_scores, cls_pred


class YOLOObjectDetector(nn.Module):
    def __init__(self, n_classes, max_n_objects, n_cells=7, predictions_per_cell=2, input_size=448,
                 backbone_name='mobilenet', conv_feature_size=3, confidence_threshold=0.1, nms_threshold=0.7,
                 feature_reduction='conv'):
        super(YOLOObjectDetector, self).__init__()
        self.logger = logging.getLogger(YOLOObjectDetector.__name__)
        self.logger.info("Creating YOLO object detector with: n_cells=%s, predictions_per_cell=%s, backbone_name=%s",
                         n_cells, predictions_per_cell, backbone_name)

        self.backbone_name = backbone_name
        self.input_size = input_size
        self.n_classes = n_classes
        self.max_n_objects = max_n_objects
        self.n_cells = n_cells
        self.predictions_per_cell = predictions_per_cell
        self.conv_feature_size = conv_feature_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.feature_reduction = feature_reduction
        if self.feature_reduction is not None and self.feature_reduction != 'mean' and self.feature_reduction != 'conv':
            raise RuntimeError("Invalid feature_reduction value: {}".format(self.feature_reduction))

        out_features = self.n_cells * self.n_cells * (self.predictions_per_cell * 5 + self.n_classes)

        if self.backbone_name == 'mobilenet':
            self.backbone = MobileNetBackbone(self.input_size)
        elif self.backbone_name == 'resnet':
            self.backbone = ResNetBackbone(self.input_size)
        else:
            raise RuntimeError("Invalid backbone name : {}".format(self.backbone_name))
        conv_map_size = int(self.backbone.conv_map_size)
        conv_map_depth = int(self.backbone.conv_map_depth)
        self.classifier = YOLODetectionModule(conv_map_size=conv_map_size,
                                              conv_map_depth=conv_map_depth,
                                              feature_reduction=self.feature_reduction,
                                              predictions_per_cell=self.predictions_per_cell,
                                              out_features=out_features,
                                              n_cells=self.n_cells,
                                              n_classes=self.n_classes)
        # TODO: Generalize
        self.spatial_scale = 1.0 / 32.0
        self.cell_size = int(self.input_size / self.n_cells)

        self.backbone = self.backbone.to(get_device())
        self.classifier = self.classifier.to(get_device())

    def forward(self, x, targets=None):
        batch_size = x.size(0)

        boxes = None
        if self.training:
            if targets is None:
                raise RuntimeError("Model is in training mode but no targets are provided!")
            else:
                boxes = targets["boxes"]

        input_h = x.size()[-2]
        input_w = x.size()[-1]
        height_factor, width_factor = self.get_scaling_factors(input_h, input_w)

        x, boxes = self.resize_down(x, height_factor, width_factor, boxes=boxes)

        conv_feature_map = self.backbone(x)
        xywh, confidence, cls_scores = self.classifier(conv_feature_map)

        if self.training:
            xywh_gt, confidence_gt, classes_gt, scores_gt = self.prepare_gt(targets)
            ious = self.get_ious(xywh, xywh_gt)
            loss = yolo_loss(xywh, confidence, cls_scores, xywh_gt, confidence_gt, classes_gt,
                             ious)

            return loss
        else:
            boxes_pred, conf_pred, cls_scores_pred = self._postprocess(xywh, confidence, cls_scores)

            labels = torch.argmax(cls_scores_pred, dim=2)

            all_boxes = []
            all_labels = []
            all_confs = []
            for i in range(batch_size):
                b = boxes_pred[i]
                l = labels[i]
                conf = conf_pred[i]

                above_threshold_idx = torch.where(conf > self.confidence_threshold)
                b = b[above_threshold_idx]
                conf = conf[above_threshold_idx]
                l = l[above_threshold_idx]

                if b.size(0) > 0:
                    b, conf, l = self.nms(b, conf, l)

                objects_found = b.size(0)
                if objects_found > self.max_n_objects:
                    topk_idx = torch.topk(conf, k=self.max_n_objects)[1]
                    b = b[topk_idx]
                    l = l[topk_idx]
                    conf = conf[topk_idx]

                b = self.resize_up(b, height_factor, width_factor)

                b = pad(b, self.max_n_objects)
                l = pad(l, self.max_n_objects)
                conf = pad(conf, self.max_n_objects)

                all_boxes.append(b)
                all_labels.append(l)
                all_confs.append(conf)

            all_boxes = torch.stack(all_boxes)
            all_labels = torch.stack(all_labels)
            all_confs = torch.stack(all_confs)

            ret = {"boxes": all_boxes, "labels": all_labels, "conf": all_confs}

            return ret

    def nms(self, boxes, confidences, cls):
        # Sort boxes according to confidence
        sorted_idx = torch.argsort(confidences, descending=True)
        idx_list = sorted_idx.tolist()

        ret_idx = []

        while len(idx_list) > 0:
            max_idx = idx_list[0]
            max_box = boxes[max_idx]
            ret_idx.append(max_idx)
            del idx_list[0]

            to_delete = []
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
        return boxes, confidences, cls

    def get_scaling_factors(self, h, w):
        height_factor = float(self.input_size) / h
        width_factor = float(self.input_size) / w
        return height_factor, width_factor

    def resize_down(self, images, height_factor, width_factor, boxes=None):
        resized_images = torch.nn.functional.interpolate(images, scale_factor=(height_factor, width_factor),
                                                         mode='bilinear')
        resized_boxes = None
        if boxes is not None:
            resized_boxes = boxes
            resized_boxes[:, :, [0, 2]] *= width_factor
            resized_boxes[:, :, [1, 3]] *= height_factor
        return resized_images, resized_boxes

    def resize_up(self, boxes, height_factor, width_factor):
        resized_boxes = boxes
        resized_boxes[:, [0, 2]] = resized_boxes[:, [0, 2]] / width_factor
        resized_boxes[:, [1, 3]] = resized_boxes[:, [1, 3]] / height_factor
        return resized_boxes

    def xywh_to_boxes(self, xywh, h, w):
        batch_size = xywh.size(0)
        xywh = xywh.clone()

        xywh[:, :, :, :, [0, 1]] = xywh[:, :, :, :, [0, 1]] * self.cell_size
        xywh[:, :, :, :, [2, 3]] = xywh[:, :, :, :, [2, 3]] * self.input_size

        for i in range(self.n_cells):
            for j in range(self.n_cells):
                xywh[:, i, j, :, 0] = xywh[:, i, j, :, 0] + i * self.cell_size
                xywh[:, i, j, :, 1] = xywh[:, i, j, :, 1] + j * self.cell_size

        xywh = xywh.view(batch_size, -1, 4)
        boxes = torch.empty(xywh.size(), device=get_device())

        boxes[:, :, 0] = xywh[:, :, 0] - 0.5 * xywh[:, :, 2]
        boxes[:, :, 1] = xywh[:, :, 1] - 0.5 * xywh[:, :, 3]
        boxes[:, :, 2] = xywh[:, :, 0] + 0.5 * xywh[:, :, 2]
        boxes[:, :, 3] = xywh[:, :, 1] + 0.5 * xywh[:, :, 3]

        boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]], min=0, max=w)
        boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]], min=0, max=h)

        return boxes

    def _postprocess(self, xywh_, confidence, cls_scores):
        batch_size = xywh_.size(0)

        xywh = xywh_.clone()
        xywh[:, :, :, :, 0] = xywh[:, :, :, :, 0] * self.cell_size
        xywh[:, :, :, :, 1] = xywh[:, :, :, :, 1] * self.cell_size

        for i in range(batch_size):
            for j in range(self.n_cells):
                for k in range(self.n_cells):
                    xywh[i, j, k, :, 0] = xywh[i, j, k, :, 0] + j * self.cell_size
                    xywh[i, j, k, :, 1] = xywh[i, j, k, :, 1] + k * self.cell_size

        xywh[:, :, :, :, 2] = xywh[:, :, :, :, 2] * self.input_size
        xywh[:, :, :, :, 3] = xywh[:, :, :, :, 3] * self.input_size

        boxes = torch.empty(batch_size, self.n_cells, self.n_cells, self.predictions_per_cell, 4)
        boxes[:, :, :, :, 0] = xywh[:, :, :, :, 0] - 0.5 * xywh[:, :, :, :, 2]
        boxes[:, :, :, :, 1] = xywh[:, :, :, :, 1] - 0.5 * xywh[:, :, :, :, 3]
        boxes[:, :, :, :, 2] = xywh[:, :, :, :, 0] + 0.5 * xywh[:, :, :, :, 2]
        boxes[:, :, :, :, 3] = xywh[:, :, :, :, 1] + 0.5 * xywh[:, :, :, :, 3]

        boxes[:, :, :, :, 0] = torch.clamp(boxes[:, :, :, :, 0], min=0, max=1241)
        boxes[:, :, :, :, 1] = torch.clamp(boxes[:, :, :, :, 1], min=0, max=374)
        boxes[:, :, :, :, 2] = torch.clamp(boxes[:, :, :, :, 2], min=0, max=1241)
        boxes[:, :, :, :, 3] = torch.clamp(boxes[:, :, :, :, 3], min=0, max=374)

        boxes = boxes.view(batch_size, -1, 4)

        # TODO: Why does this not work as single multiplication?
        cls = cls_scores.view(batch_size, self.n_cells ** 2, self.n_classes)
        c = confidence.view(batch_size, self.n_cells ** 2, self.predictions_per_cell, 1)
        cls_return = torch.empty(batch_size, self.n_cells ** 2, self.predictions_per_cell, self.n_classes)
        for i in range(c.size(0)):
            for j in range(c.size(1)):
                cls_return[i][j] = cls[i][j] * c[i][j]
        cls_return = cls_return.view(batch_size, self.n_cells ** 2 * self.predictions_per_cell, self.n_classes)
        c = c.view(batch_size, self.n_cells ** 2 * self.predictions_per_cell)
        return boxes, c, cls_return

    def prepare_gt(self, gt):
        # Prepare boxes
        boxes_gt = gt["boxes"]
        labels_gt = gt["labels"]
        ranking_scores_gt = gt["scores"]
        batch_size = boxes_gt.size(0)

        xywh = torch.zeros(batch_size, self.n_cells, self.n_cells, 4, device=get_device())
        classes = torch.zeros(batch_size, self.n_cells, self.n_cells, self.n_classes, device=get_device())
        confidence = torch.zeros(batch_size, self.n_cells, self.n_cells, 1, device=get_device())
        scores = -1 * torch.ones(batch_size, self.n_cells, self.n_cells, 1, device=get_device())

        for image_no in range(batch_size):
            image_boxes = boxes_gt[image_no]
            image_labels = labels_gt[image_no]
            image_scores = ranking_scores_gt[image_no]

            for object_no, (box, label, score) in enumerate(zip(image_boxes, image_labels, image_scores)):
                if torch.max(box) == -1 or label == -1 or score == -1:
                    continue
                    # Test
                if box[0] >= self.input_size and box[2] >= self.input_size or box[1] >= self.input_size and \
                        box[3] >= self.input_size:
                    continue

                w = box[2] - box[0]
                h = box[3] - box[1]
                x = box[0] + 0.5 * w
                y = box[1] + 0.5 * h
                cell_x = int(x // self.cell_size)
                cell_y = int(y // self.cell_size)

                # Handle boxes at the image border
                # TODO: Is this correct?
                if cell_x == self.n_cells:
                    cell_x -= 1
                if cell_y == self.n_cells:
                    cell_y -= 1

                xywh[image_no, cell_x, cell_y][0] = (x % self.cell_size) / self.cell_size
                xywh[image_no, cell_x, cell_y][1] = (y % self.cell_size) / self.cell_size
                xywh[image_no, cell_x, cell_y][2] = w / float(self.input_size)
                xywh[image_no, cell_x, cell_y][3] = h / float(self.input_size)
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

    def get_ious(self, _xywh_pred, _xywh_gt):
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

        bbox_x0_pred = xywh_pred[:, :, :, :, 0] - 0.5 * xywh_pred[:, :, :, :, 2]
        bbox_y0_pred = xywh_pred[:, :, :, :, 1] - 0.5 * xywh_pred[:, :, :, :, 3]
        bbox_x1_pred = xywh_pred[:, :, :, :, 0] + 0.5 * xywh_pred[:, :, :, :, 2]
        bbox_y1_pred = xywh_pred[:, :, :, :, 1] + 0.5 * xywh_pred[:, :, :, :, 3]

        bbox_x0_gt = xywh_gt[:, :, :, None, 0] - 0.5 * xywh_gt[:, :, :, None, 2]
        bbox_y0_gt = xywh_gt[:, :, :, None, 1] - 0.5 * xywh_gt[:, :, :, None, 3]
        bbox_x1_gt = xywh_gt[:, :, :, None, 0] + 0.5 * xywh_gt[:, :, :, None, 2]
        bbox_y1_gt = xywh_gt[:, :, :, None, 1] + 0.5 * xywh_gt[:, :, :, None, 3]

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

    def predict(self, rgb_images):
        if self.training:
            raise RuntimeError("Cannot predict in training mode")

        rgb_images = rgb_images.to(get_device())

        return self(rgb_images)

    def get_trainer(self):
        return YOLOTrainer
