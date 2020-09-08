import logging
import torch

from iorank.util.util import is_dummy_box, tensor_to_cv2_image, get_device
from mde.feature_extraction.mono_depth_extractor import RelativeHeightFeatureExtractor, \
    AtmosphericPerspectiveFeatureExtractor, ContourFeatureExtractor
from mde.feature_extraction.shapes import RectangularSegment


def boxes_to_segments(boxes, image_shape):
    """
    Turns the given bounding boxes into MDE segments.

    :param boxes: Bounding box coordinates
    :param image_shape: Shape of the input image
    :return: MDE segments
    """
    segments = [RectangularSegment(int(box[1]), int(box[3]), int(box[0]), int(box[2]), image_shape) for box in
                boxes if
                not is_dummy_box(box)]
    return segments


class HandcraftedFeatureTransformer:
    def __init__(self, **kwargs):
        """
        Creates an instance of the handcrafted feature transformer, which uses the MDE implementation for
        obtaining handcrafted features.

        :param kwargs: Keyword arguments
        """
        self.logger = logging.getLogger(HandcraftedFeatureTransformer.__name__)
        self.feature_extractors = []
        self.feature_extractors.append(RelativeHeightFeatureExtractor())
        self.feature_extractors.append(ContourFeatureExtractor())
        self.feature_extractors.append(AtmosphericPerspectiveFeatureExtractor())

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
        batch_size = all_boxes.size(0)
        padding_size = all_boxes.size(1)

        all_features = None

        for image_no in range(batch_size):
            # MDE requires cv2
            image_cv = tensor_to_cv2_image(rgb_images[image_no])

            # Turn bounding box coordinates into MDE segments
            segments = boxes_to_segments(all_boxes[image_no], image_cv.shape[:2])

            # Query the feature extractors
            features = [fe.extract_features(segments, image_cv) for fe in self.feature_extractors]
            n_features = len(self.feature_extractors)

            # Join features
            features_t = torch.ones(padding_size, n_features + 1, device=get_device())
            for i in range(len(segments)):
                f = [features[j][i] for j in range(len(self.feature_extractors))]
                f = torch.cat((torch.tensor(f).float(), torch.tensor([0.0])))
                features_t[i] = f

            if all_features is None:
                all_features = features_t.unsqueeze(0)
            else:
                all_features = torch.cat((all_features, features_t.unsqueeze(0)))

        return all_features

    def get_n_features(self):
        """
        Returns the size of the feature vectors produced by this feature transformer.

        :return: Feature vector size
        """
        return len(self.feature_extractors) + 1

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        In this case, always None is returned.

        :return: Trainer type
        """
        return None

    def set_tunable_parameters(self, **kwargs):
        self.logger.warning("No parameters to be set")
