import logging

from iorank.training.monitor import DefaultMonitor
from iorank.util.constants import metrics
from iorank.util.engine import evaluate
from iorank.util.util import get_device


class ImageObjectRankerTrainer:
    def __init__(self):
        """
        Creates a trainer for the component model.
        """

        # Do not set trainers initially (may to depend on concrete hyperparameter configuration)
        self.object_detector_trainer = None
        self.feature_transformer_trainer = None
        self.object_ranker_trainer = None
        self.object_detector_trainer_params = {}
        self.feature_transformer_trainer_params = {}
        self.object_ranker_trainer_params = {}
        self.logger = logging.getLogger(ImageObjectRankerTrainer.__name__)

    def _create_trainers(self, ioranker):
        """
        Creates and configures the trainers for the components in the given component model

        :param ioranker: Model to be trained
        """
        if ioranker.object_detector.get_trainer() is not None:
            self.object_detector_trainer = ioranker.object_detector.get_trainer()(**self.object_detector_trainer_params)
        if ioranker.feature_transformer.get_trainer() is not None:
            self.feature_transformer_trainer = ioranker.feature_transformer.get_trainer()(
                **self.feature_transformer_trainer_params)
        if ioranker.object_ranker.get_trainer() is not None:
            self.object_ranker_trainer = ioranker.object_ranker.get_trainer()(**self.object_ranker_trainer_params)

    def prepare(self, ioranker):
        """
        Prepares the trainer for the training.

        :param ioranker: Model to be trained
        :return:
        """

        self.logger.info("Prepare training")
        self._create_trainers(ioranker)

    def train(self, ioranker, dataset):
        """
        Trains the provided component model on the given dataset.

        :param ioranker: Model to be trained
        :param dataset: Training dataset
        """

        self.logger.info("Start separate training of the image object ranker components")

        object_detector = ioranker.object_detector
        feature_transformer = ioranker.feature_transformer
        object_ranker = ioranker.object_ranker

        if self.object_detector_trainer is not None:
            self.logger.info("Train object detector")
            self.object_detector_trainer.train(object_detector, dataset)
        else:
            self.logger.info("No object detector training required")

        if self.feature_transformer_trainer is not None:
            self.logger.info("Train feature transformer")
            self.feature_transformer_trainer.train(feature_transformer, dataset)
        else:
            self.logger.info("No feature transformer training required")

        if self.object_ranker_trainer is not None:
            self.logger.info("Train object ranker")
            self.object_ranker_trainer.train(object_ranker, feature_transformer, dataset)
        else:
            self.logger.info("No object ranker training required")

    def evaluate(self, model, dataset, evaluation_metrics=None):
        """
        External evaluation method, to be used for the test set.

        :param model: Model to be evaluated
        :param dataset: Test dataset
        :param evaluation_metrics: Optional. Metrics on which the model has to evaluated. If no
        metrics are provided, the model is evaluated on all implemented metrics.
        """

        def input_generator(x, y):
            x = x.to(get_device())
            return x, {}, y

        if evaluation_metrics is not None:
            m = {name: metrics[name] for name in evaluation_metrics}
        else:
            m = metrics

        monitor = DefaultMonitor(m)
        evaluate(model, dataset, input_generator, monitor)
        return monitor.get_val_score()

    def set_tunable_parameters(self, parameter_dict):
        """
        Sets tunable parameters of the trainers for the components.

        :param parameter_dict: Dict with keys in ('object_detector', 'feature_transformer', 'object_ranker') that
        contains tunable parameters for the corresponding trainers
        """
        if self.object_detector_trainer is not None and "object_detector" in parameter_dict.keys():
            self.object_detector_trainer.set_tunable_parameters(**parameter_dict["object_detector"])

        if self.feature_transformer_trainer is not None and "feature_transformer" in parameter_dict.keys():
            self.feature_transformer_trainer.set_tunable_parameters(**parameter_dict["feature_transformer"])

        if self.object_ranker_trainer is not None and "object_ranker" in parameter_dict.keys():
            self.object_ranker_trainer.set_tunable_parameters(**parameter_dict["object_ranker"])
