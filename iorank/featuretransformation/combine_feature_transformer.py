import torch

from iorank.training.combiner_trainer import CombinerTrainer


class CombineFeatureTransformer:
    def __init__(self, feature_transformers, **kwargs):
        """
        Creates an instance of the Combine Feature Transformer. It is a meta feature transformer, that combines
        features of multiple concrete transformers.

        :type feature_transformers: list
        :param feature_transformers: List of feature transformers whose features are to be combined
        :param kwargs: Keyword arguments
        """
        self.feature_transformers = feature_transformers

    def get_n_features(self):
        """
        Returns the size of the feature vectors produced by this feature transformer.

        :return: Feature vector size
        """
        n_features = 0
        # Add up number of features of individual transformers
        for ft in self.feature_transformers:
            # Dummy bit needs to be considered only once
            n_features += ft.get_n_features() - 1
        n_features += 1
        return n_features

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        :return: Trainer type
        """
        return CombinerTrainer

    def transform(self, rgb_images, all_boxes, all_labels, all_masks=None):
        """
        Produces for each of the provided images and each object a feature vector. The feature vector built by
        concatenating the feature vectors produced by the individual feature transformers.

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
        features = [ft.transform(rgb_images, all_boxes, all_labels, all_masks) for ft in self.feature_transformers]

        # Remove dummy bit for each set of feature vectors except the last one
        features_tmp = []
        for i, f in enumerate(features):
            if i != len(features) - 1:
                features_tmp.append(f[:, :, :-1])
            else:
                features_tmp.append(f)

        ret = torch.cat(features_tmp, dim=2)
        return ret

    def set_tunable_parameters(self, **kwargs):
        """
        Sets tunable parameters, which are propagated to the individual transformers.

        :param kwargs: Keyword arguments
        """
        # Tunable parameters are propagated to the individual transformers
        for ft in self.feature_transformers:
            ft.set_tunable_parameters(**kwargs)
