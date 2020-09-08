import torch
import torch.nn.functional as F

from iorank.util.util import get_device, is_dummy_box


class BoundingBoxFeatureTransformer:
    def __init__(self, **kwargs):
        """
        Creates an instance of simple feature transformer, that only takes the bounding box coordinates and image
        dimensions as features.

        :param kwargs: Keyword arguments
        """
        self.device = get_device()

    def transform(self, rgb_images, all_boxes, all_labels, all_masks=None):
        """
        Produces for each of the provided images and each object a feature vector. The feature vector only consists
        of the bounding box coordinates and the image dimensions.

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
        features = []
        for image, boxes in zip(rgb_images, all_boxes):
            # Image dimensions are required in order set bounding box coordinates in relation to the image size
            _, img_height, img_width = image.size()
            size_tensor = torch.tensor([img_height, img_width]).int()
            # Do computation on CPU
            boxes = boxes.cpu()
            image_features = []
            for box in boxes:
                # Add dummy vector for dummy bounding box
                if is_dummy_box(box):
                    image_features.append(torch.ones(self.get_n_features()).int())
                else:
                    fv = torch.cat((size_tensor, box.int()))
                    # Add dummy bit '0'
                    fv = F.pad(fv, (0, 1), mode='constant', value=0)
                    image_features.append(fv)
            features.append(torch.stack(image_features))
        result = torch.stack(features)
        # Return features on GPU if available
        result = result.to(self.device)
        return result

    def get_n_features(self):
        """
        Returns the size of the feature vectors produced by this feature transformer.

        :return: Feature vector size
        """
        # 4 bounding box coordinates + 2 image dimensions + dummy bit
        return 6 + 1

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        In this case, always None is returned.

        :return: Trainer type
        """
        # No training required
        return None
