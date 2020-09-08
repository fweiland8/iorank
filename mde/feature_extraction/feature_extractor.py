import abc


class FeatureExtractor(abc.ABC):
    """
    Components extracting features from given segments and images (considered as contexts).
    """
    @abc.abstractmethod
    def extract_features(self, segments, image):
        pass
