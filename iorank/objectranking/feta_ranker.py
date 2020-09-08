import logging
import numpy as np
import torch

from csrank import FETAObjectRanker
from iorank.training.object_ranker_trainer import ObjectRankerTrainer
from iorank.util.util import get_device


class FETARanker:
    def __init__(self, n_objects, n_object_features, add_zeroth_order_model=False, n_hidden=2, n_units=8,
                 reg_strength=1e-4, learning_rate=1e-3, batch_size=128, **kwargs):
        """
        Creates an instance of the FETA (First Evaluate Then Aggregate) object ranker. This class uses the FETA
        ranker from the csrank library.

        :param n_objects: Number of objects to be ranked (can be seen as upper bound as padding is used)
        :param n_object_features: Size of the feature vectors
        :param add_zeroth_order_model: If True, the (context-independent) 0-order model is taken into account.
        Default: False
        :param n_hidden: Number of hidden layers. Default: 8
        :param n_units: Number of hidden units. Default: 2
        :param reg_strength: Regularization strength of the regularize function. Default: 1e-4
        :param learning_rate: Learning rate used for training. Default: 1e-3
        :param batch_size: Batch size used for training. Default: 128
        :param kwargs: Keyword arguments
        """

        self.model = None
        self.n_objects = n_objects
        self.n_object_features = n_object_features
        self.torch_model = False
        self.trainable = True
        self.device = get_device()
        self.logger = logging.getLogger(FETARanker.__name__)

        # Tunable parameters
        self.add_zeroth_order_model = add_zeroth_order_model
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.reg_strength = reg_strength
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self._construct_model()

    def _construct_model(self):
        """
        Constructs the FETA model.

        """
        self.logger.info("Construct model..")
        self.model = FETAObjectRanker(self.n_objects, self.n_object_features,
                                      add_zeroth_order_model=self.add_zeroth_order_model)
        self.model.set_tunable_parameters(n_hidden=self.n_hidden, n_units=self.n_units, reg_strength=self.reg_strength,
                                          learning_rate=self.learning_rate, batch_size=self.batch_size)
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

    def set_tunable_parameters(self, add_zeroth_order_model=False, n_hidden=2, n_units=8,
                               reg_strength=1e-4, learning_rate=1e-3, batch_size=128):
        """
        Set the tunable parameters for this model.

        :param add_zeroth_order_model: If True, the (context-independent) 0-order model is taken into account.
        Default: False
        :param n_hidden: Number of hidden layers. Default: 8
        :param n_units: Number of hidden units. Default: 2
        :param reg_strength: Regularization strength of the regularize function. Default: 1e-4
        :param learning_rate: Learning rate used for training. Default: 1e-3
        :param batch_size: Batch size used for training. Default: 128
        :return:
        """
        self.add_zeroth_order_model = add_zeroth_order_model
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.reg_strength = reg_strength
        self.learning_rate = learning_rate
        self.batch_size = batch_size

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
