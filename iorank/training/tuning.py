import logging
import numpy as np
import time
from datetime import datetime
from skopt import Optimizer
from torch.utils.data import Subset

from iorank.datasets.util import get_split
from iorank.util.util import get_root_dataset


class HyperparameterTuner:
    def __init__(self, trainer, model, model_parameter_ranges, training_parameter_ranges, time_limit, n_iterations=30,
                 cv_metric='kendalls_tau'):
        """
        Creates a hyperparameter tuner for optimizing hyperparameters of the component model.

        :param trainer: Trainer object to be used
        :param model: Model to be trained
        :param model_parameter_ranges: Ranges for the model hyperparameters
        :param training_parameter_ranges: Ranges for the trainer hyperparameters
        :param time_limit: Time limit for hyperparameter optimization (in seconds)
        :param n_iterations: Number of tuning iterations. Default: 30
        :param cv_metric: Metric for evaluating a cross-validation result. Default: 'kendalls_tau'
        """

        self.logger = logging.getLogger(HyperparameterTuner.__name__)
        self.trainer = trainer
        self.model = model
        self.n_iterations = n_iterations
        self.time_limit = time_limit
        self.cv_metric = cv_metric
        self.model_parameter_ranges = model_parameter_ranges
        self.model_parameter_map = None
        self.training_parameter_ranges = training_parameter_ranges
        self.training_parameter_map = None
        self.optimizer = None

        self.logger.info("Model param ranges are: %s", self.model_parameter_ranges)
        self.logger.info("Trainer param ranges are: %s", self.training_parameter_ranges)

        # Compute deadline (reserve 5h for training on whole dataset + evaluation)
        current_time = int(time.time())
        self.deadline = current_time + self.time_limit - 18000
        self.logger.info("Deadline is %s", datetime.fromtimestamp(self.deadline).strftime("%d.%m.%Y %H:%M:%S"))

        # Initial estimate for how long an iteration takes
        self.iteration_durations = [7200]

    def _prepare_optimizer(self):
        """
        Prepares the sk-optimize optimizer. Creates an optimizer parameter space containing the model and trainer
        hyperparameters.

        """
        counter = 0
        optimizer_parameter_space = []
        self.model_parameter_map = {}
        self.training_parameter_map = {}
        for component_name in self.model_parameter_ranges.keys():
            self.model_parameter_map[component_name] = {}
            for parameter_name in self.model_parameter_ranges[component_name].keys():
                self.model_parameter_map[component_name][parameter_name] = counter
                counter += 1
                optimizer_parameter_space.append(self.model_parameter_ranges[component_name][parameter_name])
        for component_name in self.training_parameter_ranges.keys():
            self.training_parameter_map[component_name] = {}
            for parameter_name in self.training_parameter_ranges[component_name].keys():
                self.training_parameter_map[component_name][parameter_name] = counter
                counter += 1
                optimizer_parameter_space.append(self.training_parameter_ranges[component_name][parameter_name])
        self.optimizer = Optimizer(optimizer_parameter_space)

    def _set_model_parameters(self, parameters):
        """
        Sets the hyperparameters of the individual model components.

        :param parameters: Dict containing the model hyperparameters
        """
        for component_name in self.model_parameter_ranges.keys():
            component = getattr(self.model, component_name)
            parameter_dict = {}

            for parameter_name in self.model_parameter_ranges[component_name]:
                idx = self.model_parameter_map[component_name][parameter_name]
                parameter_dict[parameter_name] = parameters[idx]
            self.logger.info("Model parameters for %s: %s", component_name, parameter_dict)
            if len(parameter_dict) > 0:
                component.set_tunable_parameters(**parameter_dict)

    def _set_trainer_parameters(self, parameters):
        """
        Sets the hyperparameters of the individual component trainers.

        :param parameters: Dict containing the trainer hyperparameters
        """
        parameter_dict = {}
        for component_name in self.training_parameter_ranges.keys():
            parameter_dict[component_name] = {}
            for parameter_name in self.training_parameter_ranges[component_name]:
                idx = self.training_parameter_map[component_name][parameter_name]
                parameter_dict[component_name][parameter_name] = parameters[idx]
        self.logger.info("Trainer parameters are: %s", parameter_dict)
        self.trainer.set_tunable_parameters(parameter_dict)

    def tune(self, dataset):
        """
        Tunes the model this tuner has been created for on the given dataset.

        :param dataset: Training data
        :return The tuned model
        """

        self.logger.info("Start hyperparameter optimization")
        self._prepare_optimizer()

        best_validation_score = 0.0
        best_parameters = None

        for iter_no in range(self.n_iterations):
            self.logger.info("Start iteration %s/%s", iter_no + 1, self.n_iterations)

            # Check time
            start_time = int(time.time())
            if start_time + np.mean(self.iteration_durations) > self.deadline:
                self.logger.warning("No time left for another iteration")
                break

            parameters = self.optimizer.ask()
            self._set_model_parameters(parameters)

            # Do not finetune object detector in HP optimization to save time
            self.model.object_detector.set_tunable_parameters(finetuning=False)

            # Prepare => Create trainers
            self.trainer.prepare(self.model)

            # Set hyperparameters of the trainers
            self._set_trainer_parameters(parameters)

            self.model.prepare()

            validation_scores = []

            n_folds = 2
            split = get_split(dataset)
            indices = split(dataset, 0.5, shuffle=True)

            for fold_no in range(n_folds):
                self.logger.info("Start CV fold %s/%s", fold_no + 1, n_folds)

                # Construct train and val set
                if fold_no == 0:
                    train_indices, val_indices = indices[0], indices[1]
                elif fold_no == 1:
                    train_indices, val_indices = indices[1], indices[0]
                root_dataset = get_root_dataset(dataset)
                dataset_train = Subset(root_dataset, train_indices)
                dataset_val = Subset(root_dataset, val_indices)

                self.logger.debug("Train on %s instances, validate on %s instances", len(dataset_train),
                                  len(dataset_val))
                self.trainer.train(self.model, dataset_train)
                self.logger.info("Finished training, evaluating..")
                validation_result = self.trainer.evaluate(self.model, dataset_val,
                                                          evaluation_metrics=[self.cv_metric])
                score = validation_result[self.cv_metric]
                self.logger.info("CV Fold %s/%s has score %s", fold_no + 1, n_folds, score)
                validation_scores.append(score)
            validation_score = np.mean(validation_scores)
            self.logger.info("Score of the chosen parameters is %s", validation_score)
            self.optimizer.tell(parameters, validation_score)
            if validation_score > best_validation_score:
                self.logger.info("Score is new best score!")
                best_validation_score = validation_score
                best_parameters = parameters

            iteration_duration = int(time.time()) - start_time
            self.logger.info("Iteration took %s seconds", iteration_duration)
            self.iteration_durations.append(iteration_duration)
            self.logger.info("Estimate for the next iteration is %s", np.mean(self.iteration_durations))

        # Train model on best parameters found
        self.logger.info("Train model on best parameters found")
        self._set_model_parameters(best_parameters)
        self._set_trainer_parameters(best_parameters)

        # Enable finetuning for final prediction
        self.model.object_detector.set_tunable_parameters(finetuning=True)

        self.model.prepare()
        self.trainer.prepare(self.model)
        self.trainer.train(self.model, dataset)
        self.logger.info("Finished hyperparameter optimization")
        return self.model
