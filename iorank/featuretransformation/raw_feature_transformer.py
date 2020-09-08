from iorank.featuretransformation.visual_appearance_feature_transformer import VisualAppearanceFeatureTransformer
from iorank.util.util import get_device


class RawFeatureTransformer(VisualAppearanceFeatureTransformer):
    def __init__(self, input_type='default', reduced_size=64, use_masks=False, **kwargs):
        """
        Creates an instance of the raw feature transformer, which takes the raw RGB data of an image path as feature
        vector.

        :param input_type: Input type for the auto encoder. If 'default', then an object region is cut out.
            If 'blacked', then the image is blacked except for the object region. Default: 'default'
        :param reduced_size: Size to which inputs are scaled before being processed. Default: 64
        :param use_masks: If True, object masks are applied to the image patches before they are processed. Default: False
        :param kwargs: Keyword arguments
        """
        super(RawFeatureTransformer, self).__init__()
        self.device = get_device()
        self.use_masks = use_masks

        # Tunable parameters
        self.input_type = input_type
        self.reduced_size = reduced_size

    def get_features_for_input(self, inputs):
        """
        Constructs for the given inputs (patches) feature vectors.

        :param inputs: Image patches showing objects
        :return: Feature vectors
        """
        return inputs.flatten(1)

    def get_n_features(self):
        """
        Returns the size of the feature vectors produced by this feature transformer.

        :return: Feature vector size
        """
        return self.reduced_size ** 2 * 3 + 1

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        In this case, always None is returned.

        :return: Trainer type
        """
        return None
