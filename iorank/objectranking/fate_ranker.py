import logging
import numpy as np
import torch

from csrank.objectranking.fate_object_ranker import FATEObjectRanker
from iorank.training.object_ranker_trainer import ObjectRankerTrainer
from iorank.util.util import get_device


class FATERanker:
    def __init__(self, n_object_features, n_hidden_set_units=32, n_hidden_set_layers=2, n_hidden_joint_units=32,
                 n_hidden_joint_layers=2, reg_strength=1e-4, learning_rate=1e-3,
                 batch_size=128, **kwargs):
        """
        Creates an instance of the FATE (First Aggregate Then Evaluate) object ranker. This class uses the FATE
        ranker from the csrank library.

        :param n_object_features: Size of the feature vectors
        :param n_hidden_set_units: Number of units in the hidden set layers. Default: 32
        :param n_hidden_set_layers: Number of hidden set layers. Default: 2
        :param n_hidden_joint_units: Number of units in the hidden joint layers. Default: 32
        :param n_hidden_joint_layers: Number of hidden joint layers. Default: 2
        :param reg_strength: Regularization strength of the regularize function. Default: 1e-4
        :param learning_rate: Learning rate used for training. Default: 1e-3
        :param batch_size: Batch size used for training. Default: 128
        :param kwargs: Keyword arguments
        """

        self.model = None
        self.n_object_features = n_object_features
        self.torch_model = False
        self.trainable = True
        self.device = get_device()
        self.logger = logging.getLogger(FATERanker.__name__)

        # Tunable parameters
        self.n_hidden_set_units = n_hidden_set_units
        self.n_hidden_set_layers = n_hidden_set_layers
        self.n_hidden_joint_units = n_hidden_joint_units
        self.n_hidden_joint_layers = n_hidden_joint_layers
        self.reg_strength = reg_strength
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self._construct_model()

    def _construct_model(self):
        """
        Constructs the actual FATE model.

        """
        self.logger.info("Construct model..")
        self.model = FATEObjectRanker(self.n_object_features)
        self.model.set_tunable_parameters(n_hidden_set_units=self.n_hidden_set_units,
                                          n_hidden_set_layers=self.n_hidden_set_layers,
                                          n_hidden_joint_units=self.n_hidden_joint_units,
                                          n_hidden_joint_layers=self.n_hidden_joint_layers,
                                          reg_strength=self.reg_strength, learning_rate=self.learning_rate,
                                          batch_size=self.batch_size)
        self.logger.info("Finished constructing model")

    def fit(self, X, Y, **kwargs):
        """
        Fits the model on the given data.

        :param X: Examples
        :param Y: Ground truth data
        :param kwargs: Keyword arguments
        """

        # Turn PyTorch tensors into numpy array first
        X = np.array(X)
        Y = np.array(Y)
        self.model.fit(X, Y, **kwargs)

    def predict_scores(self, object_feature_vectors, **kwargs):
        """
        Predict utility scores for object ranking for the given feature vectors.

        :param object_feature_vectors: Object feature vectors
        :param kwargs: Keyword arguments
        :return: Utility scores
        """
        # Turn PyTorch tensors into numpy array first
        fv = np.array(object_feature_vectors)
        return torch.tensor(self.model.predict_scores(fv), device=self.device)

    def set_tunable_parameters(self, n_hidden_set_units=32, n_hidden_set_layers=2, n_hidden_joint_units=32,
                               n_hidden_joint_layers=2, reg_strength=1e-4, learning_rate=1e-3,
                               batch_size=128):
        """
        Sets the tunable parameters of this model.

        :param n_hidden_set_units: Number of units in the hidden set layers. Default: 32
        :param n_hidden_set_layers: Number of hidden set layers. Default: 2
        :param n_hidden_joint_units: Number of units in the hidden joint layers. Default: 32
        :param n_hidden_joint_layers: Number of hidden joint layers. Default: 2
        :param reg_strength: Regularization strength of the regularizer function. Default: 1e-4
        :param learning_rate: Learning rate used for training. Default: 1e-3
        :param batch_size: Batch size used for training. Default: 128
        """
        self.n_hidden_set_units = n_hidden_set_units
        self.n_hidden_set_layers = n_hidden_set_layers
        self.n_hidden_joint_units = n_hidden_joint_units
        self.n_hidden_joint_layers = n_hidden_joint_layers
        self.reg_strength = reg_strength
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Reconstruct model with new parameters
        self._construct_model()

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        :return: Trainer type for this model
        """
        return ObjectRankerTrainer

    def set_n_object_features(self, n_object_features):
        self.n_object_features = n_object_features
        # Reconstruct model as number of features might have changed
        self._construct_model()
