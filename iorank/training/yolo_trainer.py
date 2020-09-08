import os

import logging
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset, DataLoader

from iorank.datasets.util import get_split
from iorank.metrics.metrics import object_detection_recall
from iorank.util.names import optimizers
from iorank.util.util import get_device, get_root_dataset, get_uuid


class YOLOTrainingMonitor:
    def __init__(self):
        """
        Creates a monitor for YOLO training.

        """
        self.metric_values_val = []
        self.loss_values = []

    def add_train_result(self, losses):
        """
        This method is called for each training batch.

        :param losses:  A tensor of loss values
        """
        self.loss_values.extend(losses.tolist())

    def add_val_result(self, pred, y):
        """
        This method is called for each validation batch.

        :param pred: Predicted data
        :param y: Ground truth data
        """
        self.metric_values_val.append(object_detection_recall(pred, y))

    def get_train_summary(self):
        """
        Returns a summary of the achieved training results.

        :return: A string with the average loss in training
        """
        mean_loss = np.mean(self.loss_values)
        ret = "loss = {}".format(mean_loss)
        return ret

    def get_val_summary(self):
        """
        Returns a summary of the achieved validation results.

        :return: A string with the average metric value in validation
        """
        val_score = self.get_val_score()
        return "val_score = {:0.4f}".format(val_score)

    def get_val_score(self):
        """
        Returns a score describing the performance on the validation set.

        In this case the validation score is given as the average metric value.

        :return: Validation score
        """
        val_score = np.mean(self.metric_values_val)
        return val_score

    def reset(self):
        """
        This method is called after each epoch in order to reset the results.

        """
        self.metric_values_val = []
        self.loss_values = []

    def is_better_than(self, best_val_result):
        """
        Compares the provided result (from a preceding epoch) to the achieved results in the current epoch.

        :param best_val_result: Validation result to which the current results are to be compared
        :return: True, if the current result is better than the provided result, False otherwise
        """
        if best_val_result is None:
            return True
        val_score = self.get_val_score()
        # Higher is better
        return val_score > best_val_result


class YOLOTrainer:
    def __init__(self, max_n_epochs=20,
                 optimizer="Adam", lr=1e-4, batch_size=16, lr_scheduler=None, early_stopping=True,
                 early_stopping_patience=3):
        """
        Creates a trainer for training the YOLO object detector.

        :param max_n_epochs: Maximum number of training epochs. Default: 20
        :param optimizer: Optimizer to be used for training. Default: Adam
        :param lr: Initial learning rate. Default: 1e-4
        :param batch_size: Batch size for training. Default: 16
        :param lr_scheduler: Function for learning rate scheduling. If no function is provided, the learning rate
        is not scheduled.
        :param early_stopping: If True, early stopping is done. Default: True
        :param early_stopping_patience: Number of epoch without improvement before training is stopped. Default: 3
        """

        self.logger = logging.getLogger(YOLOTrainer.__name__)
        self.max_n_epochs = max_n_epochs
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience

    def _evaluate_internal(self, detector, dataset, monitor):
        """
        Internal evaluation method used for evaluating the performance on the validation set.

        :param detector: Object detector to be evaluated
        :param dataset: Dataset to be considered
        :param monitor: Monitor object to which the results are given
        """
        data_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True)
        for i, data in enumerate(data_loader, 0):
            self.logger.debug("Start evaluation batch %s", i + 1)
            x, y = data
            x = x.to(get_device())
            with torch.no_grad():
                pred = detector(x)
            monitor.add_val_result(pred, y)
            self.logger.debug("Finished evaluation batch %s", i + 1)

    def _train_one_epoch(self, detector, dataloader, monitor, optimizer):
        """
        Performs a single training epoch.

        :param detector: Object detector to be trained
        :param dataloader: Dataloader providing the training data
        :param monitor: Monitor object to which the training results are given
        :param optimizer: Optimizer for training
        """
        for i, data in enumerate(dataloader, 0):
            self.logger.debug("Start training batch %s", i + 1)
            x, y = data
            x = x.to(get_device())
            loss = detector(x, targets=y)
            optimizer.zero_grad()
            self.logger.debug("Doing gradient step")
            loss.mean().backward()
            optimizer.step()
            monitor.add_train_result(loss)
            self.logger.debug("Finished training batch %s", i + 1)

    def train(self, yolo_detector, dataset):
        """
        Trains the provided YOLO object detector on the given dataset.

        :param yolo_detector: Model to be trained
        :param dataset: Training dataset
        :return:
        """

        # Assign a lower learning rate to the backbone
        backbone_params = [param for param in yolo_detector.backbone.parameters() if param.requires_grad]
        classifier_params = [param for param in yolo_detector.classifier.parameters() if param.requires_grad]
        params = [{'name': 'classifier', 'params': classifier_params, 'lr': self.lr},
                  {'name': 'backbone', 'params': backbone_params, 'lr': self.lr * 0.1}]
        opt = optimizers[self.optimizer](params, weight_decay=1e-5)

        # Start learning rate decay after 15 epochs
        def lr_scheduler(epoch):
            if epoch < 15:
                lr_factor = 1
            else:
                lr_factor = max(0.95 ** (epoch - 15), 0.1)
            return lr_factor

        scheduler = LambdaLR(opt, lr_lambda=lr_scheduler)

        monitor = YOLOTrainingMonitor()

        dataset_full_size = len(dataset)
        self.logger.info("Start batch learning on dataset with %s instances", dataset_full_size)
        self.logger.info("Train up to %s epochs", self.max_n_epochs)
        val_split = 0.2

        split = get_split(dataset)
        train_indices, val_indices = split(dataset, val_split, shuffle=True)
        root_dataset = get_root_dataset(dataset)
        dataset_train = Subset(root_dataset, train_indices)
        dataset_val = Subset(root_dataset, val_indices)

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
            yolo_detector.train()
            self._train_one_epoch(yolo_detector, data_loader_train, monitor, opt)
            self.logger.info("[Epoch %s/%s]: training result: %s", t + 1, self.max_n_epochs,
                             monitor.get_train_summary())
            yolo_detector.eval()
            self._evaluate_internal(yolo_detector, dataset_val, monitor)
            self.logger.info("[Epoch %s/%s]: validation result: %s", t + 1, self.max_n_epochs,
                             monitor.get_val_summary())

            if not self.early_stopping:
                self.logger.info("Early stopping disabled, finishing epoch")
                continue

            if monitor.is_better_than(best_val_result):
                self.logger.info("Validation result is new best result, saving model..")
                best_val_result = monitor.get_val_score()
                torch.save(yolo_detector.state_dict(), checkpoint_file)
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
            yolo_detector.load_state_dict(torch.load(checkpoint_file))
            # Cleanup
            os.remove(checkpoint_file)
