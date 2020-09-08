import logging
import torch
from torch import nn
from torch.nn import ReLU, Module

from iorank.featuretransformation.visual_appearance_feature_transformer import VisualAppearanceFeatureTransformer
from iorank.training.autoencoder_trainer import AutoEncoderTrainer
from iorank.util.util import get_device


class AutoEncoderConvolutional(Module):
    def __init__(self, n_latent_features=1024, reduced_size=64, activation=ReLU()):
        """
        Creates a convolutional auto encoder that can be used for feature transformation.

        :type activation: function
        :param n_latent_features: Number of neurons in the bottleneck layer. Default: 1024
        :param reduced_size: Size to which inputs are scaled before being processed. Default: 64
        :param activation: Activation function
        """

        super().__init__()
        self.logger = logging.getLogger(AutoEncoderConvolutional.__name__)
        self.n_latent_features = n_latent_features
        self.reduced_size = reduced_size
        self.middle_layer_size = int(16 * self.reduced_size / 4 * self.reduced_size / 4)
        self.activation = activation

        self.logger.info("Construct model..")
        self.encode_conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.encode_pool1 = nn.MaxPool2d(2, stride=2)
        self.encode_conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.encode_pool2 = nn.MaxPool2d(2, stride=2)
        self.encode_fc = nn.Linear(self.middle_layer_size, self.n_latent_features)

        self.decode_fc = nn.Linear(self.n_latent_features, self.middle_layer_size)
        self.decode_conv1 = nn.ConvTranspose2d(16, 6, kernel_size=2, stride=2)
        self.decode_conv2 = nn.ConvTranspose2d(6, 3, kernel_size=2, stride=2)

        self.reset_weights()

        self.logger.info("Finished instantiation")

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encode_conv1(x)
        x = self.activation(x)
        x = self.encode_pool1(x)
        x = self.encode_conv2(x)
        x = self.activation(x)
        x = self.encode_pool2(x)
        x = torch.flatten(x, 1)
        x = self.encode_fc(x)
        return x

    def decode(self, x):
        x = self.decode_fc(x)
        x = self.activation(x)
        x = x.view(-1, 16, int(self.reduced_size / 4), int(self.reduced_size / 4))
        x = self.activation(x)
        x = self.decode_conv1(x)
        x = self.activation(x)
        x = self.decode_conv2(x)
        x = self.activation(x)
        return x

    def reset_weights(self):
        # Reset weights of all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class AutoEncoderFeatureTransformer(VisualAppearanceFeatureTransformer):
    def __init__(self, n_latent_features=1024, reduced_size=64, input_type='default', use_masks=False, **kwargs):
        """
        Creates an instance of the Auto Encoder Feature Transformer. It generates a feature vector for a detection by
        forwarding its image patch through the encode part auf an auto encoder.

        :param n_latent_features: Number of neurons in the bottleneck layer. Default: 1024
        :param reduced_size: Size to which inputs are scaled before being processed. Default: 64
        :param input_type: Input type for the auto encoder. If 'default', then an object region is cut out.
            If 'blacked', then the image is blacked except for the object region. Default: 'default'
        :param use_masks: If True, object masks are applied to the image patches before they are processed.
        Default: False
        :param kwargs: Keyword arguments
        """
        super(AutoEncoderFeatureTransformer, self).__init__()
        self.device = get_device()
        self.use_masks = use_masks

        # Tunable parameters
        self.n_latent_features = n_latent_features
        self.reduced_size = reduced_size
        self.input_type = input_type

        self.logger = logging.getLogger(AutoEncoderFeatureTransformer.__name__)

        self._construct_model()

    def _construct_model(self):
        """
        Constructs the actual auto encoder and moves it to CUDA if available.

        """
        self.model = AutoEncoderConvolutional(self.n_latent_features, self.reduced_size)
        self.model = self.model.to(self.device, non_blocking=True)

    def __call__(self, x, **kwargs):
        """
        Processes the given input.

        :param x: Model input
        :param kwargs: Keyword arguments
        :return: The model output for the input
        """
        return self.model(x)

    def parameters(self):
        """
        Returns the model parameters.

        :return: Model parameters
        """
        return self.model.parameters()

    def get_n_features(self):
        """
        Returns the size of the feature vectors produced by this feature transformer.

        :return: Feature vector size
        """
        # +1 due to dummy bit
        return self.model.n_latent_features + 1

    def get_features_for_input(self, inputs):
        """
        Constructs for the given inputs (patches) feature vectors.

        :param inputs: Image patches showing objects
        :return: Feature vectors
        """
        inputs = inputs.to(self.device)
        with torch.no_grad():
            features = self.model.encode(inputs)
        return features

    def get_reduced_size(self):
        return self.model.reduced_size

    def reset(self):
        self.model.reset_weights()

    def set_tunable_parameters(self, n_latent_features=1024, reduced_size=64, input_type='default', **kwargs):
        """
        Sets tunable parameters for this model.

        :param n_latent_features: Number of neurons in the bottleneck layer. Default: 1024
        :param reduced_size: Size to which inputs are scaled before being processed. Default: 64
        :param input_type: Input type for the auto encoder. If 'default', then an object region is cut out.
            If 'blacked', then the image is blacked except for the object region. Default: 'default'
        :param kwargs: Keyword arguments
        """

        self.logger.info("Set parameters: n_latent_features=%s, reduced_size=%s, input_type=%s", n_latent_features,
                         reduced_size, input_type)

        self.n_latent_features = n_latent_features
        self.reduced_size = reduced_size
        self.input_type = input_type

        self._construct_model()

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        :return: Trainer type
        """
        return AutoEncoderTrainer

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
