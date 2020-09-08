import logging


class CombinerTrainer:
    def __init__(self):
        """
        Creates an instance of the Combiner Trainer, which trains the combine feature transformer. This
        trainer is only responsible for starting the training of the individual feature transformers.
        """

        self.logger = logging.getLogger(CombinerTrainer.__name__)
        self.tunable_parameters = None

    def train(self, combiner, dataset):
        """
        Trains the provided combiner on the given dataset.

        :param combiner: The combiner, whose individual feature transformers are to be trained
        :param dataset: The dataset on which training takes place
        """
        feature_transformers = combiner.feature_transformers
        self.logger.info("Train CombineFeatureTransformer consisting of %s individual transformers",
                         len(feature_transformers))
        for transformer in combiner.feature_transformers:
            trainer_type = transformer.get_trainer()
            # No trainer type returned => No training required
            if trainer_type is not None:
                trainer = trainer_type()
                # Set trainer tunable parameters
                if self.tunable_parameters is not None:
                    trainer.set_tunable_parameters(**self.tunable_parameters)
                self.logger.info("Training %s", trainer.__class__.__name__)
                trainer.train(transformer, dataset)
            else:
                self.logger.info("No training required for %s", transformer.__class__.__name__)

    def set_tunable_parameters(self, **kwargs):
        """
        Sets tunable parameters for trainers. These parameters are simply propagated to the individual feature
        transformer trainers.

        :param kwargs: Keyword arguments (trainer parameters)

        """
        self.logger.info("Parameters: %s", kwargs)
        self.tunable_parameters = kwargs
