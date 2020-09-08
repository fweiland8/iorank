import torch
import torch.nn as nn
from torchvision import models

from iorank.featuretransformation.visual_appearance_feature_transformer import VisualAppearanceFeatureTransformer
from iorank.util.util import get_device


class AlexNetFeatureTransformer(VisualAppearanceFeatureTransformer):
    def __init__(self, input_type='default', reduced_size=64, use_masks=False, **kwargs):
        """
        Creates an instance of the AlexNet feature transformer. A pretrained AlexNet image classification network is
        used in order to produce features for object region patches.

        :param input_type: Input type for AlexNet. If 'default', then an object region is cut out.
            If 'blacked', then the image is blacked except for the object region. Default: 'default'
        :param reduced_size: Size to which inputs are scaled before being processed. Default: 64
        :param use_masks: If True, object masks are applied to the image patches before they are processed. Default: False
        :param kwargs: Keyword arguments
        """
        super(AlexNetFeatureTransformer, self).__init__()
        self.device = get_device()
        self.model = None
        self.use_masks = use_masks

        # Tunable parameters
        self.input_type = input_type
        self.reduced_size = reduced_size

        self._construct_model()

    def _construct_model(self):
        """
        Constructs the model and moves it to CUDA if available.

        """
        model = models.alexnet(pretrained=True)
        self.model = nn.Sequential(
            *list(model.children())[:-1]
        )
        self.model.eval()
        self.model = self.model.to(self.device)

    def get_n_features(self):
        """
        Returns the size of the feature vectors produced by this feature transformer.

        :return: Feature vector size
        """
        # +1 due to dummy bit
        return 9216 + 1

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        In this case, always None is returned.

        :return: Trainer type
        """
        # Pretrained model => No training required
        return None

    def get_features_for_input(self, inputs):
        """
        Constructs for the given inputs (patches) feature vectors.

        :param inputs: Image patches showing objects
        :return: Feature vectors
        """
        inputs = inputs.to(self.device)
        with torch.no_grad():
            features = self.model(inputs)
        return features


class ResNetFeatureTransformer(VisualAppearanceFeatureTransformer):
    def __init__(self, input_type='default', reduced_size=64, use_masks=False, **kwargs):
        """
        Creates an instance of the ResNet feature transformer. A pretrained ResNet image classification network is
        used in order to produce features for object region patches.

        :param input_type: Input type for ResNet. If 'default', then an object region is cut out.
            If 'blacked', then the image is blacked except for the object region. Default: 'default'
        :param reduced_size: Size to which inputs are scaled before being processed. Default: 64
        :param use_masks: If True, object masks are applied to the image patches before they are processed. Default: False
        :param kwargs: Keyword arguments
        """
        super(ResNetFeatureTransformer, self).__init__()
        self.device = get_device()
        self.model = None
        self.use_masks = use_masks

        # Tunable parameters
        self.input_type = input_type
        self.reduced_size = reduced_size

        self._construct_model()

    def _construct_model(self):
        """
        Constructs the model and moves it to CUDA if available.

        """
        r = models.resnet18(pretrained=True)
        self.model = nn.Sequential(
            *list(r.children())[:-1]
        )
        self.model.eval()
        self.model = self.model.to(self.device)

    def get_n_features(self):
        """
        Returns the size of the feature vectors produced by this feature transformer.

        :return: Feature vector size
        """
        # +1 due to dummy bit
        return 512 + 1

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        In this case, always None is returned.

        :return: Trainer type
        """
        # Pretrained model => No training required
        return None

    def get_features_for_input(self, inputs):
        """
        Constructs for the given inputs (patches) feature vectors.

        :param inputs: Image patches showing objects
        :return: Feature vectors
        """
        inputs = inputs.to(self.device)
        with torch.no_grad():
            features = self.model(inputs)
        return features
