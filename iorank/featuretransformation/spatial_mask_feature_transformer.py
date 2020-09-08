import logging
import torch

from iorank.util.util import is_dummy_box, get_device, get_spatial_mask


class SpatialMaskFeatureTransformer:
    def __init__(self, mask_size=64, **kwargs):
        """
        Creates an instance of the spatial mask feature transformer. It creates a spatial mask for each bounding box,
        describing its position and extent. The bounding box is finally downscaled in order to decrease the number of
        features. A feature vector is obtained by flattening the spatial mask.

        :param mask_size: Side length of the downscaled mask. Default: 64
        :param kwargs: Keyword arguments
        """
        self.logger = logging.getLogger(SpatialMaskFeatureTransformer.__name__)
        self.mask_size = mask_size

    def get_n_features(self):
        """
        Returns the size of the feature vectors produced by this feature transformer.

        :return: Feature vector size
        """
        return self.mask_size ** 2 + 1

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        In this case, always None is returned.

        :return: Trainer type
        """
        # No training required
        return None

    def transform(self, rgb_images, all_boxes, all_labels, all_masks=None):
        """
        Produces for each of the provided images and each object a feature vector. The feature vector is a
        flattened representation of a spatial mask for a bounding box.

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
        h, w = rgb_images.size()[-2:]
        padding_size = all_boxes.size(1)
        features = None
        for image, boxes in zip(rgb_images, all_boxes):
            image_features = torch.empty(padding_size, self.mask_size ** 2 + 1, device=get_device())
            for box_no, box in enumerate(boxes):
                # Use dummy vector consisting of only 1's
                if is_dummy_box(box):
                    dummy_mask = torch.ones(self.mask_size ** 2 + 1, device=get_device())
                    image_features[box_no] = dummy_mask.float()
                else:
                    mask = get_spatial_mask(box, h, w, mask_size=self.mask_size)
                    mask = torch.cat((mask, torch.tensor([0.0], device=get_device())))
                    image_features[box_no] = mask

            if features is None:
                features = image_features.unsqueeze(0)
            else:
                features = torch.cat((features, image_features.unsqueeze(0)))
        return features

    def set_tunable_parameters(self, mask_size=64, **kwargs):
        """
        Sets tunable parameters for this model.

        :param mask_size: Side length of the downscaled mask. Default: 64
        :param kwargs: Keyword arguments
        """
        self.logger.info("Set parameters: mask_size=%s", mask_size)
        self.mask_size = mask_size
