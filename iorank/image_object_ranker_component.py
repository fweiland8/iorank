import logging

from iorank.util.util import expand_boxes


class ImageObjectRankerComponent:
    def __init__(self, object_detector, feature_transformer, object_ranker, use_masks=False, box_expansion_factor=0,
                 **kwargs):
        """
        Creates an image object ranker (component architecture) that uses the delivered components.

        :param object_detector: Object detector component
        :param feature_transformer: Feature transformer component
        :param object_ranker: Object ranker component
        :param use_masks: If True, object masks are used for feature transformation. Default: False
        :param box_expansion_factor: Factor by which bounding boxes can be expanded in order to
        add additional context information. Default: 0 (no expansion)
        :param kwargs: Keyword arguments
        """
        self.object_detector = object_detector
        self.feature_transformer = feature_transformer
        self.object_ranker = object_ranker
        self.use_masks = use_masks
        self.box_expansion_factor = box_expansion_factor
        self.logger = logging.getLogger(ImageObjectRankerComponent.__name__)

    def __call__(self, rgb_images):
        """
        Produces a prediction for the given RGB images.

        N: Batch size

        :param rgb_images: Tensor of RGB images of size (N,3,H,W)
        :return: Dict with the predictions for bounding boxes, class labels, confidence scores and utility scores.
        Each prediction result is given as tensor.
        """

        # 1. Object detection
        self.logger.debug("Starting object detection")
        object_detection_pred = self.object_detector.predict(rgb_images)
        self.logger.debug("Done object detection")
        self.logger.debug("Starting feature transformation (use_masks = %s)", self.use_masks)

        masks = None
        if self.use_masks:
            masks = object_detection_pred["masks"]

        if self.box_expansion_factor != 0:
            self.logger.debug("Expand boxes (factor = %s)", self.box_expansion_factor)
            image_height, image_width = rgb_images.size()[-2:]
            boxes_for_feature_transformation = expand_boxes(object_detection_pred["boxes"], self.box_expansion_factor,
                                                            image_height,
                                                            image_width)
        else:
            boxes_for_feature_transformation = object_detection_pred["boxes"]

        # 2. Feature Transformation
        features = self.feature_transformer.transform(rgb_images, boxes_for_feature_transformation,
                                                      object_detection_pred["labels"], masks)

        self.logger.debug("Done feature transformation")

        # 3. Object ranking
        self.logger.debug("Starting object ranking")
        if not self.object_ranker.torch_model:
            features = features.cpu()
        scores = self.object_ranker.predict_scores(features, images=rgb_images)
        self.logger.debug("Done object ranking")

        target = {"boxes": object_detection_pred["boxes"],
                  "labels": object_detection_pred["labels"],
                  "conf": object_detection_pred["conf"],
                  "scores": scores}

        if self.use_masks:
            target["masks"] = object_detection_pred["masks"]

        return target

    def prepare(self):
        """
        Prepares the image object ranker for usage.

        It is checked if the feature vector size expected by the object ranker fits to the number of features produced
        by the feature transformation component. If not, the object ranker is reconstructed with the correct
        feature vector size.
        """
        self.logger.info("Prepare model")

        # Feature Vector size must be set correctly
        if hasattr(self.object_ranker, 'n_object_features'):
            object_ranker_fv_size = self.object_ranker.n_object_features
            if object_ranker_fv_size != self.feature_transformer.get_n_features():
                self.logger.warning(
                    "FV size of feature transformer and object ranker do not match. Reconstruct object ranker..")
                self.object_ranker.set_n_object_features(self.feature_transformer.get_n_features())
