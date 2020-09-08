import torch


def get_mean_y_for_box(box, img_height):
    """
    Returns the average y-coordinate of the given bounding box. The value is returned relative to the image height.

    :param box: Bounding box coordinates of the form (x0,y0,x1,y1)
    :param img_height: Height of the image the bounding box occurs in
    :return: The average y-coordinate of the box, in relation to the image height
    """
    y_abs = box[1] + 0.5 * (box[3] - box[1])
    y_rel = y_abs / img_height
    return y_rel


class BaselineRanker:
    def __init__(self, **kwargs):
        """
        Creates an instance of the baseline ranker. The baseline ranker ranks objects according to their average
        y-coordinate.

        :param kwargs:
        """
        self.torch_model = False
        self.trainable = False

    def fit(self, X, Y, **kwargs):
        """
        Fits the model on the given data.

        Unused in this case.

        :param X: Examples
        :param Y: Ground truth data
        :param kwargs: Keyword arguments
        """
        # Method is not use as no training takes place
        return

    def predict_scores(self, object_feature_vectors, **kwargs):
        """
        Predict utility scores for object ranking for the given feature vectors.

        :param object_feature_vectors: Object feature vectors
        :param kwargs: Keyword arguments
        :return: Utility scores
        """
        # Assumption: Feature vectors are bounding box coordinates + [height, width]
        n_images = object_feature_vectors.shape[0]

        results = []
        for i in range(n_images):
            # Score = Average y-coordinate
            scores = [get_mean_y_for_box(fv[0:4], fv[5]) for fv in object_feature_vectors[i]]
            results.append(torch.stack(scores))

        return torch.stack(results)

    def get_trainer(self):
        """
        Returns a trainer object for this model. Unused for the baseline ranker.

        :return: Trainer type for this model
        """
        # No training required
        return None
