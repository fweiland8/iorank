import logging
import torch
from torch import nn
from torchvision import models

from iorank.training.object_ranker_trainer import ObjectRankerTrainer
from iorank.util.util import get_device
from torchcsrank.core.feta_network import FETANetwork


class TorchFETARanker:
    def __init__(self, n_object_features, n_hidden_layers=2, n_hidden_units=8, use_image_context=False, **kwargs):
        """
        Creates an instance of the FETA (First Evaluate Then Aggregate) object ranker. This class uses an implementation
        based on PyTorch.

        :param n_object_features: Size of the feature vectors
        :param n_hidden_layers: Number of hidden layers. Default: 2
        :param n_hidden_units: Number of hidden units. Default: 8
        :param use_image_context: If True, an image representative is used as additional context information. Default: False
        :param kwargs: Keyword arguments
        """

        self.logger = logging.getLogger(TorchFETARanker.__name__)
        self.n_object_features = n_object_features
        self.use_image_context = use_image_context
        self.model = None
        self.device = get_device()
        self.torch_model = True
        self.trainable = True

        # Tunable parameters
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.use_image_context = use_image_context

        self._construct_model()

    def _construct_model(self):
        """
        Constructs the FETA model and the optional model for the image representation.

        """
        self.logger.info("Construct model..")
        n_context_features = 0
        if self.use_image_context:
            # 512 is the number of features produced by ResNet18
            n_context_features = 512
        self.model = FETANetwork(self.n_object_features, n_hidden_layers=self.n_hidden_layers,
                                 n_hidden_units=self.n_hidden_units, n_context_features=n_context_features)
        self.logger.info("Move model to device %s", self.device)
        self.model = self.model.to(self.device, non_blocking=True)

        self.logger.info("Create image representation model")
        resnet = models.resnet18(pretrained=True)
        self.image_representation_model = nn.Sequential(
            *list(resnet.children())[:-1]
        )
        self.image_representation_model.eval()
        self.image_representation_model = self.image_representation_model.to(self.device)

        self.logger.info("Finished instantiation")

    def __call__(self, X, **kwargs):
        """
        Processes the given input feature vectors.

        :param X: Input feature vectors
        :param kwargs: Keyword arguments
        :return: Utility scores
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X).float()

        image_contexts = None
        if self.use_image_context:
            if "images" in kwargs.keys():
                images = kwargs["images"]
                image_contexts = self.image_representation_model(images)
                image_contexts = image_contexts.flatten(1)
            else:
                raise RuntimeError("Image context should be used but not images are provided!")

        return self.model(X, image_contexts=image_contexts)

    def predict_scores(self, X, **kwargs):
        """
        Predict utility scores for object ranking for the given feature vectors.

        :param X: Object feature vectors
        :param kwargs: Keyword arguments
        :return: Utility scores
        """
        pred = self(X, **kwargs)
        return pred.detach()

    def parameters(self):
        """
        Returns the model parameters.

        :return: Model parameters
        """
        return self.model.parameters()

    def reset(self):
        self.logger.info("Reset model..")
        self.model.reset_weights()

    def set_tunable_parameters(self, n_hidden_layers=2, n_hidden_units=8, use_image_context=False):
        """
        Sets the tunable parameters for this model.

        :param n_hidden_layers: Number of hidden layers. Default: 2
        :param n_hidden_units: Number of hidden units. Default: 8
        :param use_image_context: If True, an image representative is used as additional context information. Default: False
        """
        self.logger.info("Set parameters: n_hidden_layers=%s, n_hidden_units=%s, use_image_context=%s", n_hidden_units,
                         n_hidden_units,
                         use_image_context)
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.use_image_context = use_image_context

        self._construct_model()

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        :return: Trainer type for this model
        """
        return ObjectRankerTrainer

    def set_n_object_features(self, n_object_features):
        self.n_object_features = n_object_features
        self._construct_model()

    def train(self):
        self.model.train()
        self.image_representation_model.train()

    def eval(self):
        self.model.eval()
        self.image_representation_model.eval()
