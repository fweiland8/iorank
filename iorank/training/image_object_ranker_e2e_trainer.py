import os

import logging
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset, DataLoader

from iorank.datasets.util import get_split
from iorank.metrics.metric_utils import harmonic_mean
from iorank.metrics.metrics import kendalls_tau, reduced_ranking, object_detection_recall
from iorank.training.monitor import DefaultMonitor
from iorank.util.constants import metrics
from iorank.util.engine import evaluate
from iorank.util.names import optimizers
from iorank.util.util import get_device, get_root_dataset, get_uuid


class EndToEndTrainingMonitor:
    def __init__(self):
        """
        Creates a monitor for training the E2E model.

        """
        self.metric_values_val_od = []
        self.metric_values_val_or = []
        self.loss_values_od = []
        self.loss_values_or = []

    def _get_metric_value_od(self, pred, y):
        """
        Evaluates the object detection performance for the given data.
        
        :param pred: Predicted data
        :param y: Ground truth data
        :return: Metric value for object detection
        """

        return object_detection_recall(pred, y)

    def _get_metric_value_or(self, pred, y):
        """
        Evaluates the object ranking performance for the given data.
        
        :param pred: Predicted data
        :param y: Ground truth data
        :return: Metric value for object ranking 
        """
        return reduced_ranking(kendalls_tau)(pred, y)

    def add_train_result(self, od_loss, r_loss):
        """
        This method is called for each training batch.

        :param od_loss: Loss value of the object detection loss function
        :param r_loss: Loss value of the object ranking loss function
        """

        # Append the loss values to the lists
        self.loss_values_od.extend(od_loss.tolist())
        self.loss_values_or.extend(r_loss.tolist())

    def add_val_result(self, pred, y):
        """
        This method is called for each validation batch.
        
        :param pred: Predicted data
        :param y: Ground truth data
        """

        # Compute metric for detection and ranking
        self.metric_values_val_od.append(self._get_metric_value_od(pred, y))
        self.metric_values_val_or.append(self._get_metric_value_or(pred, y))

    def get_train_summary(self):
        """
        Returns a summary over the achieved training results.

        :return: A string with the average loss in training 
        """

        mean_od_loss = np.mean(self.loss_values_od)
        mean_or_loss = np.mean(self.loss_values_or)
        mean_total_loss = mean_od_loss + mean_or_loss
        ret = "loss = {} (od_loss = {}, or_loss = {})".format(mean_total_loss, mean_od_loss, mean_or_loss)
        return ret

    def get_val_summary(self):
        """
        Returns a summary over the achieved validation results.

        :return: A string with the average score in validation
        """
        detection_score = np.mean(self.metric_values_val_od)
        ranking_score = np.mean(self.metric_values_val_or)
        val_score = self.get_val_score()
        return "detection score = {:0.4f}, ranking score = {:0.4f}, val_score = {:0.4f}".format(detection_score,
                                                                                                ranking_score,
                                                                                                val_score)

    def get_val_score(self):
        """
        Returns a score describing the performance on the validation set.
        
        In this case, the validation score is given as the harmonic mean of 
        the detection and ranking scores.

        :return: Validation score
        """

        od_mean = np.mean(self.metric_values_val_od)
        or_mean = np.mean(self.metric_values_val_or)
        val_score = harmonic_mean(od_mean, or_mean)
        return val_score

    def reset(self):
        """
        This method is called after each epoch in order to reset the results.

        """
        self.metric_values_val_od = []
        self.metric_values_val_or = []
        self.loss_values_od = []
        self.loss_values_or = []

    def is_better_than(self, best_val_result):
        """
        Compares the provided result (from a preceding epoch) to the achieved results in the current epoch.

        :param best_val_result: Validation result to which the current results are to be compared
        :return: True, if the current result is better than the provided result, False otherwise
        """
        if best_val_result is None:
            return True
        val_score = self.get_val_score()
        #  Compare the validation scores, higher is better
        return val_score > best_val_result


class ImageObjectRankerE2ETrainer:
    def __init__(self, loss_function=None, max_n_epochs=20,
                 optimizer="Adam", lr=1e-4, batch_size=16, lr_scheduler=None, early_stopping=False,
                 early_stopping_patience=3):
        """
        Creates a trainer for the E2E model.

        :param loss_function: Loss function to be used. If no loss function is provided, the sum
        of detection loss and ranking loss is taken
        :param max_n_epochs: Maximum number of training epochs. Default: 20
        :param optimizer: Optimizer to be used for training. Default: Adam
        :param lr: Initial learning rate. Default: 1e-4
        :param batch_size: Batch size for training. Default: 16
        :param lr_scheduler: Function for learning rate scheduling. If no function is provided, the learning rate
        is not scheduled.
        :param early_stopping: If True, early stopping is done. Default: False
        :param early_stopping_patience: Number of epoch without improvement before training is stopped. Default: 3
        """

        self.logger = logging.getLogger(ImageObjectRankerE2ETrainer.__name__)
        self.max_n_epochs = max_n_epochs
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.loss_function = loss_function
        if self.loss_function is None:
            self.logger.warning("Using default loss function")

            # Default loss: Simple sum
            def loss(od_loss, r_loss):
                return od_loss + r_loss

            self.loss_function = loss

    def _evaluate_internal(self, ioranker, dataset, monitor):
        """
        Internal evaluation method used for evaluating the performance on the validation set.
        
        :param ioranker: Ranker to be evaluated
        :param dataset: Dataset to be considered
        :param monitor: Monitor object to which the results are given
        """

        data_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True)
        for i, data in enumerate(data_loader, 0):
            self.logger.debug("Start evaluation batch %s", i + 1)
            x, y = data
            x = x.to(get_device())
            with torch.no_grad():
                pred = ioranker(x)
            monitor.add_val_result(pred, y)
            self.logger.debug("Finished evaluation batch %s", i + 1)

    def _train_one_epoch(self, ioranker, dataloader, monitor, optimizer):
        """
        Performs a single training epoch.

        :param ioranker: Ranker to be trained
        :param dataloader: Dataloader providing the training data
        :param monitor: Monitor object to which the training results are given
        :param optimizer: Optimizer for training
        """
        for i, data in enumerate(dataloader, 0):
            self.logger.debug("Start training batch %s", i + 1)
            x, y = data
            x = x.to(get_device())
            od_loss, r_loss = ioranker(x, targets=y)
            loss = self.loss_function(od_loss, r_loss)
            optimizer.zero_grad()
            self.logger.debug("Doing gradient step")
            loss.mean().backward()
            optimizer.step()
            monitor.add_train_result(od_loss, r_loss)
            self.logger.debug("Finished training batch %s", i + 1)

    def train(self, ioranker, dataset):
        """
        Trains the provided E2E ranker on the given dataset.

        :param ioranker: Model to be trained
        :param dataset: Training dataset
        """

        # Init optimizer
        # Use different learning rates for the individual modules
        backbone_params = [param for param in ioranker.backbone.parameters() if param.requires_grad]
        classifier_params = [param for param in ioranker.classifier.parameters() if param.requires_grad]
        ranker_params = [param for param in ioranker.ranker.parameters() if param.requires_grad]
        params = [{'name': 'classifier', 'params': classifier_params, 'lr': self.lr},
                  {'name': 'backbone', 'params': backbone_params, 'lr': self.lr * 0.1},
                  {'name': 'ranker', 'params': ranker_params, 'lr': self.lr}]
        opt = optimizers[self.optimizer](params, weight_decay=1e-5)

        # Init scheduler
        scheduler = None
        if self.lr_scheduler is not None:
            scheduler = LambdaLR(opt, lr_lambda=self.lr_scheduler)

        # Init montior
        monitor = EndToEndTrainingMonitor()

        # Prepare dataset
        dataset_full_size = len(dataset)
        val_split = 0.2
        split = get_split(dataset)
        train_indices, val_indices = split(dataset, val_split, shuffle=True)
        root_dataset = get_root_dataset(dataset)
        dataset_train = Subset(root_dataset, train_indices)
        dataset_val = Subset(root_dataset, val_indices)

        self.logger.info("Start batch learning on dataset with %s instances", dataset_full_size)
        self.logger.info("Train up to %s epochs", self.max_n_epochs)
        self.logger.info("Training on %s instances, validating on %s instances", len(dataset_train), len(dataset_val))
        data_loader_train = DataLoader(dataset_train, batch_size=self.batch_size, pin_memory=True)

        checkpoint_file = "checkpoint_{}.pt".format(get_uuid())
        self.logger.debug("Write checkpoints to %s", checkpoint_file)

        patience_counter = 0
        best_val_result = None

        # Do actual training
        for t in range(self.max_n_epochs):
            monitor.reset()
            self.logger.debug("Start epoch %s", t + 1)
            ioranker.train()
            self._train_one_epoch(ioranker, data_loader_train, monitor, opt)
            self.logger.info("[Epoch %s/%s]: training result: %s", t + 1, self.max_n_epochs,
                             monitor.get_train_summary())
            ioranker.eval()
            self._evaluate_internal(ioranker, dataset_val, monitor)
            self.logger.info("[Epoch %s/%s]: validation result: %s", t + 1, self.max_n_epochs,
                             monitor.get_val_summary())

            if not self.early_stopping:
                self.logger.info("Early stopping disabled, finishing epoch")
                continue

            if monitor.is_better_than(best_val_result):
                self.logger.info("Validation result is new best result, saving model..")
                best_val_result = monitor.get_val_score()
                torch.save(ioranker.state_dict(), checkpoint_file)
                patience_counter = 0
            else:
                if patience_counter < self.early_stopping_patience:
                    patience_counter += 1
                    self.logger.info("Validation result did not improve, patience counter is %s", patience_counter)
                else:
                    self.logger.info("Stopping training since model does not improve")
                    break

            if scheduler is not None:
                scheduler.step()

        if self.early_stopping and os.path.exists(checkpoint_file):
            ioranker.load_state_dict(torch.load(checkpoint_file))
            # Cleanup
            os.remove(checkpoint_file)

    def evaluate(self, model, dataset):
        """
        External evaluation method, to be used for the test set.

        :param model: Model to be evaluated
        :param dataset: Test dataset
        :return:
        """

        def input_generator(x, y):
            x = x.to(get_device())
            return x, {}, y

        monitor = DefaultMonitor(metrics)
        evaluate(model, dataset, input_generator, monitor)
        return monitor.get_val_score()
