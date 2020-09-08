import logging
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from iorank.training.faster_rcnn_trainer import FasterRCNNTrainer
from iorank.util.util import get_device, pad


class FasterRCNN:
    def __init__(self, n_classes, max_n_objects=10, confidence_threshold=0.1, finetuning=True):
        """
        Creates an instance of the Faster R-CNN object detector.

        :param n_classes: Number of classes in the considered dataset
        :param max_n_objects: Maximum number of objects. If more objects are detected, only the detection with the
        highest confidence are retained. Default: 10
        :param confidence_threshold: Confidence value a detection must have in order to be retained. Default: 0.1
        :param finetuning: If True, the object detector is finetuned to the considered dataset. Especially, the box
        predictor is replaced with a new one that has the desired number of classes.
        """

        self.logger = logging.getLogger(FasterRCNN.__name__)
        self.logger.info(
            "Instantiating object ranker (max_n_objects=%s, confidence_threshold=%s)",
            max_n_objects, confidence_threshold)
        self.confidence_threshold = confidence_threshold
        self.max_n_objects = max_n_objects
        self.n_classes = n_classes
        self.finetuning = finetuning

        self.device = get_device()

        self._construct_model()

    def _construct_model(self):
        self.logger.info("Construct model..")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                          box_detections_per_img=self.max_n_objects,
                                                                          box_score_thresh=self.confidence_threshold)

        if self.finetuning:
            self.model.roi_heads.box_predictor = FastRCNNPredictor(1024, self.n_classes)

        self.logger.info("Move to device %s", self.device)
        self.model = self.model.to(self.device, non_blocking=True)
        self.logger.info("Finished instantiation")

    def predict(self, rgb_images):
        self.model.eval()
        self.logger.debug("Start prediction..")

        self.logger.debug("Move input to device %s", self.device)
        rgb_images = rgb_images.to(self.device, non_blocking=True)
        self.logger.debug("Finished moving to device")

        self.logger.debug("Call model..")
        result = self.model(rgb_images)
        self.logger.debug("Finished calling model")

        self.logger.debug("Postprocessing model output")

        # Apply padding in order to return detections as tensor
        boxes = []
        labels = []
        confidences = []
        for i in range(len(rgb_images)):
            b = result[i]["boxes"]
            b = pad(b, self.max_n_objects)
            boxes.append(b)
            l = result[i]["labels"]
            l = pad(l, self.max_n_objects)
            labels.append(l)
            c = result[i]["scores"]
            c = pad(c, self.max_n_objects)
            confidences.append(c)

        # Turn detections to tensor
        boxes = torch.stack(boxes)
        labels = torch.stack(labels)
        confidences = torch.stack(confidences)

        ret = {"boxes": boxes, "labels": labels, "conf": confidences}

        self.logger.debug("Finished prediction")
        return ret

    def get_trainer(self):
        """
        Returns a trainer type for this model. Only in case the detector is to be finetuned, a trainer is returned.

        :return: Trainer type for this model
        """
        # Only return trainer if detector is to be finetuned
        if self.finetuning:
            return FasterRCNNTrainer
        else:
            return None

    def set_tunable_parameters(self, finetuning=True, **kwargs):
        self.logger.info("Set parameters: finetuning=%s", finetuning)
        self.finetuning = finetuning

        self._construct_model()
