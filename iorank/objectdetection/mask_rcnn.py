import logging
import torch
import torchvision

from iorank.util.util import get_device, pad


class MaskRCNN:
    def __init__(self, max_n_objects=10, confidence_threshold=0.7, **kwargs):
        """
        Creates an instance of the Mask R-CNN image segmentation model.

        :param max_n_objects: Maximum number of objects. If more objects are detected, only the detection with the
        highest confidence are retained. Default: 10
        :param confidence_threshold: Confidence value a detection must have in order to be retained. Default: 0.7
        """

        self.logger = logging.getLogger(MaskRCNN.__name__)
        self.confidence_threshold = confidence_threshold
        self.max_n_objects = max_n_objects
        self.device = get_device()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                        box_detections_per_img=max_n_objects,
                                                                        box_score_thresh=confidence_threshold)

        self.model.eval()

        self.logger.info("Move to device %s", self.device)
        self.model = self.model.to(self.device, non_blocking=True)
        self.logger.info("Finished instantiation")

    def predict(self, rgb_images):
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
        masks = []
        confidences = []
        labels = []
        for i in range(len(rgb_images)):
            b = result[i]["boxes"]
            b = pad(b, self.max_n_objects)
            boxes.append(b)

            # So far, the mask consists of scores in [0,1]. Thus, the mask needs to be turned into a binary mask first.
            m = result[i]["masks"]
            m = m.squeeze(1)
            for j in range(m.size(0)):
                m[j] = torch.where(m[j] > 0.5, torch.ones(m[j].size(), device=self.device),
                                   torch.zeros(m[j].size(), device=self.device))
            m = pad(m, self.max_n_objects)
            masks.append(m)

            c = result[i]["scores"]
            c = pad(c, self.max_n_objects)
            confidences.append(c)

            l = torch.ones(result[i]["scores"].size(0), device=self.device)
            l = pad(l, self.max_n_objects)
            labels.append(l)

        boxes = torch.stack(boxes)
        masks = torch.stack(masks)
        confidences = torch.stack(confidences)
        labels = torch.stack(labels)

        ret = {"boxes": boxes, "masks": masks, "conf": confidences, "labels": labels}

        self.logger.debug("Finished prediction")
        return ret

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        :return: Trainer type for this model
        """
        # No finetuning takes place for the Mask R-CNN model => No trainer required
        return None
