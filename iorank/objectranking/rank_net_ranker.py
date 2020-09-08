import numpy as np
import torch

from csrank import RankNet
from iorank.training.object_ranker_trainer import ObjectRankerTrainer
from iorank.util.util import get_device


class RankNetRanker:
    def __init__(self, n_object_features, **kwargs):
        """
        Creates an instance of the RankNet ranker.

        :param n_object_features: Size of the feature vectors
        :param kwargs: Keyword arguments
        """
        self.model = None
        self.n_object_features = n_object_features
        self.torch_model = False
        self.trainable = True
        self.device = get_device()

        self._construct_model()

    def _construct_model(self):
        """
        Constructs the RankNet model.

        """
        self.model = RankNet(self.n_object_features)

    def fit(self, X, Y, **kwargs):
        """
        Fits the model on the given data.

        :param X: Examples
        :param Y: Ground truth data
        :param kwargs: Keyword arguments
        """
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
        fv = np.array(object_feature_vectors)
        return torch.tensor(self.model.predict_scores(fv), device=self.device)

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        :return: Trainer type for this model
        """
        return ObjectRankerTrainer

    def set_n_object_features(self, n_object_features):
        self.n_object_features = n_object_features
        self._construct_model()
