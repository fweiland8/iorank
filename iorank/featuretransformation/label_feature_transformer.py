import logging
import torch

from iorank.util.util import get_device


class LabelFeatureTransformer:
    def __init__(self, n_classes=1, **kwargs):
        """
        Creates an instance of the label feature transformer. The produced feature vector is a one-hot encoded
        version of an object's class label.

        :param n_classes: Number of classes in the dataset
        :param kwargs: Keyword arguments
        """
        self.n_classes = n_classes
        self.logger = logging.getLogger(LabelFeatureTransformer.__name__)
        self.logger.info("Created label feature transformer for %s classes", self.n_classes)

    def get_n_features(self):
        """
        Returns the size of the feature vectors produced by this feature transformer.

        :return: Feature vector size
        """
        # +1 due to dummy bit
        return self.n_classes + 1

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        In this case, always None is returned.

        :return: Trainer type
        """
        # No training is required
        return None

    def transform(self, rgb_images, all_boxes, all_labels, all_masks=None):
        """
        Produces for each of the provided images and each object a feature vector. The feature vector is a one-hot
        encoded version of an object's class label.

        N : Batch size \n
        U: Padding size (upper bound for the number of objects) \n
        H: Image height \n
        W: Image width \n
        F: Number of produced features

        :param rgb_images: Tensor of RGB images of size (N,3,H,W)
        :param all_boxes: Tensor of bounding box coordinates of size (N,U,4)
        :param all_labels: Tensor of class labels of size (N,U)
        :param all_masks: Optional: Tensor of object masks of size (N,U,H,W)
        :return: Feature matrix of size (N,U,F)
        """
        n_images = all_boxes.size(0)
        padding_size = all_boxes.size(1)
        ret = torch.ones(n_images, padding_size, self.get_n_features(), device=get_device())
        for i in range(n_images):
            for j in range(padding_size):
                if all_labels[i][j] != -1:
                    ret[i, j] = torch.zeros(self.get_n_features())
                    ret[i, j, all_labels[i][j]] = 1
        return ret
