import torch


class RandomRanker:
    def __init__(self, **kwargs):
        """
        Creates an instance of a simple random ranker, that ranks the objects in the input set randomly.

        :param kwargs: Keyword arguments
        """
        self.torch_model = False
        self.trainable = False

    def predict_scores(self, object_feature_vectors, **kwargs):
        """
        Predict utility scores for object ranking for the given feature vectors.

        :param object_feature_vectors: Object feature vectors
        :param kwargs: Keyword arguments
        :return: Utility scores
        """
        n_images = object_feature_vectors.shape[0]
        n_objects = object_feature_vectors.shape[1]

        results = [torch.randperm(n_objects) for _ in range(n_images)]

        return torch.stack(results)

    def get_trainer(self):
        """
        Returns a trainer type for this model.

        In this case, always None is returned.

        :return: Trainer type for this model
        """
        # No training required
        return None
