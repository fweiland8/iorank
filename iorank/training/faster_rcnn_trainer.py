import os

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from iorank.datasets.util import get_split
from iorank.metrics.metric_utils import harmonic_mean
from iorank.metrics.metrics import object_detection_precision, label_accuracy, \
    object_detection_recall, reduced_ranking
from iorank.util.util import get_uuid, get_root_dataset, get_device, pad


class RCNNTrainingMonitor:
    def __init__(self):
        """
        Creates a monitor for Faster R-CNN training / finetuning.

        """
        self.loss_values = {}
        self.metric_values = {
            "precision": [],
            "recall": [],
            "label_accuracy": []
        }

    def add_train_result(self, losses_dict):
        """
        This method is called for each training batch.

        :param losses_dict: Dict with loss names and loss values
        """
        for loss_name in losses_dict.keys():
            if loss_name not in self.loss_values.keys():
                self.loss_values[loss_name] = []
            self.loss_values[loss_name].append(float(losses_dict[loss_name]))

    def add_val_result(self, pred, y):
        """
        This method is called for each validation batch.

        :param pred: Predicted data
        :param y: Ground truth data
        """
        self.metric_values["precision"].append(object_detection_precision(pred, y))
        self.metric_values["recall"].append(object_detection_recall(pred, y))
        self.metric_values["label_accuracy"].append(reduced_ranking(label_accuracy)(pred, y))

    def get_train_summary(self):
        """
        Returns a summary of the achieved training results.

        :return: A string with the training performance
        """
        mean_losses = {loss_name: np.mean(losses) for loss_name, losses in self.loss_values.items()}
        train_score = np.mean(list(mean_losses.values()))

        metric_strings = ["{} = {:0.4f}".format(loss_name, mean_loss) for loss_name, mean_loss in
                          mean_losses.items()]
        ret = ",".join(metric_strings)
        ret += ", train_score = {:0.4f}".format(train_score)

        return ret

    def get_val_summary(self):
        """
        Returns a summary of the achieved validation results.

        :return: A string with the validation performance
        """
        precision_mean = np.mean(self.metric_values["precision"])
        recall_mean = np.mean(self.metric_values["recall"])
        label_acc_mean = np.mean(self.metric_values["label_accuracy"])
        val_score = self.get_val_score()
        return "avg precision = {:0.4f}, avg recall = {:0.4f}, avg label accuracy = {:0.4f}, val_score = {:0.4f}".format(
            precision_mean,
            recall_mean,
            label_acc_mean,
            val_score)

    def get_val_score(self):
        """
        Returns a score describing the performance on the validation set.

        In this case, the score is given as the harmonic mean of precision, recall and label accuracy.

        :return: Validation score
        """
        precision_mean = np.mean(self.metric_values["precision"])
        recall_mean = np.mean(self.metric_values["recall"])
        label_acc_mean = np.mean(self.metric_values["label_accuracy"])
        val_score = harmonic_mean(precision_mean, recall_mean, label_acc_mean)
        return val_score

    def reset(self):
        """
        This method is called after each epoch in order to reset the results.

        """
        self.loss_values = {}
        self.metric_values = {
            "precision": [],
            "recall": [],
            "label_accuracy": []
        }

    def is_better_than(self, best_val_result):
        """
        Compares the provided result (from a preceding epoch) to the achieved results in the current epoch.

        :param best_val_result: Validation result to which the current results are to be compared
        :return: True, if the current result is better than the provided result, False otherwise
        """

        if best_val_result is None:
            return True
        val_score = self.get_val_score()
        # Compare the harmonic means, higher is better
        return val_score > best_val_result


def input_generator(x, y):
    """
    Input generator for Faster R-CNN training. Brings the ground truth in the required format. Especially, padding
    is removed.

    :param x: Input image
    :param y: Ground truth data
    :return: Input data and ground truth data for training
    """
    x = x.to(get_device())
    targets = []
    for i in range(len(y["boxes"])):
        boxes = y["boxes"][i]
        # Remove padding
        boxes = boxes[boxes > -1].view(-1, 4)
        boxes = boxes.to(device=get_device(), dtype=torch.int)
        labels = y["labels"][i]
        # REmove padding
        labels = labels[labels > -1]
        labels = labels.to(get_device())
        targets.append({"boxes": boxes, "labels": labels})
    return x, {}, targets


def postprocess_prediction(pred, max_ranking_size):
    """
    Postprocesses the prediction by Faster R-CNN in order to apply the standard metric methods. Especially,
    the predictions need to be padded to the provided maximum ranking size and stacked into a single tensor.

    :param pred: Predicted data coming from Faster R-CNN
    :param max_ranking_size: Maximum number of objects to be considered (padding size)
    :return: Postprocessed, padded predicted data
    """
    all_boxes = []
    all_labels = []
    all_confs = []
    for i in range(len(pred)):
        boxes = pred[i]["boxes"]
        boxes = pad(boxes, max_ranking_size)
        all_boxes.append(boxes)

        labels = pred[i]["labels"]
        labels = pad(labels, max_ranking_size)
        all_labels.append(labels)

        confs = pred[i]["scores"]
        confs = pad(confs, max_ranking_size)
        all_confs.append(confs)

    all_boxes = torch.stack(all_boxes)
    all_labels = torch.stack(all_labels)
    all_confs = torch.stack(all_confs)

    return {"boxes": all_boxes, "labels": all_labels, "conf": all_confs}


class FasterRCNNTrainer:
    def __init__(self, max_n_epochs=10, lr=5e-3, batch_size=8, early_stopping=True,
                 early_stopping_patience=3, **kwargs):
        """
        Creates a trainer for training / finetuning a Faster R-CNN object detector.

        :param max_n_epochs: Maximum number of training epochs. Default: 10
        :param lr: Initial learning rate. Default: 1e-5
        :param batch_size: Batch size for training. Default: 8
        :param early_stopping: If True, early stopping is done. Default: True
        :param early_stopping_patience: Number of epoch without improvement before training is stopped. Default: 3
        :param kwargs: Keyword arguments
        """
        self.logger = logging.getLogger(FasterRCNNTrainer.__name__)
        self.max_n_epochs = max_n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience

    def _train_one_epoch(self, model, dataloader, monitor, optimizer):
        """
        Performs a single training epoch.

        :param model: Model to be trained
        :param dataloader: Dataloader providing the training data
        :param monitor: Monitor object to which the training results are given
        :param optimizer: Optimizer for training
        """
        for i, data in enumerate(dataloader, 0):
            self.logger.debug("Start training batch %s", i + 1)
            x, y_orig = data
            x, kwargs, y = input_generator(x, y_orig)
            losses_dict = model(x, **kwargs, targets=y)
            # Total loss is sum of the individual losses
            loss = torch.sum(torch.stack(list(losses_dict.values())))
            monitor.add_train_result(losses_dict)
            optimizer.zero_grad()
            self.logger.debug("Doing gradient step")
            loss.backward()
            optimizer.step()
            self.logger.debug("Finished training batch %s", i + 1)

    def _evaluate_internal(self, rcnn_model, data_loader, monitor):
        """
        Internal evaluation method used for evaluating the performance on the validation set.

        :param rcnn_model: Model to be evaluated
        :param data_loader: Data loader for the validation data
        :param monitor: Monitor object to which the results are given
        :return:
        """
        for i, data in enumerate(data_loader, 0):
            self.logger.debug("Start evaluation batch %s", i + 1)
            x, y = data
            x, _, _ = input_generator(x, y)
            with torch.no_grad():
                pred = rcnn_model(x)
            pred = postprocess_prediction(pred, 22)
            monitor.add_val_result(pred, y)
            self.logger.debug("Finished evaluation batch %s", i + 1)

    def train(self, object_detector, dataset):
        """
        Trains the provided object detector on the given dataset.

        :param object_detector: Model to be trained
        :param dataset: Training dataset
        """

        # Extracts the actual PyTorch model from the wrapper class
        rcnn_model = object_detector.model

        params = rcnn_model.parameters()

        opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

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
        data_loader_val = DataLoader(dataset_val, batch_size=16, pin_memory=True)

        monitor = RCNNTrainingMonitor()
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 3, gamma=0.1)
        best_val_result = None

        checkpoint_file = "checkpoint_{}.pt".format(get_uuid())
        self.logger.debug("Write checkpoints to %s", checkpoint_file)

        patience_counter = 0
        for t in range(self.max_n_epochs):
            monitor.reset()
            self.logger.debug("Start epoch %s", t + 1)
            rcnn_model.train()
            self._train_one_epoch(rcnn_model, data_loader_train, monitor, opt)
            self.logger.info("[Epoch %s/%s]: training result: %s", t + 1, self.max_n_epochs,
                             monitor.get_train_summary())
            rcnn_model.eval()
            self._evaluate_internal(rcnn_model, data_loader_val, monitor)
            self.logger.info("[Epoch %s/%s]: validation result: %s", t + 1, self.max_n_epochs,
                             monitor.get_val_summary())

            scheduler.step()

            if not self.early_stopping:
                self.logger.info("Early stopping disabled, finishing epoch")
                continue

            if monitor.is_better_than(best_val_result):
                self.logger.info("Validation result is new best result, saving model..")
                best_val_result = monitor.get_val_score()
                torch.save(rcnn_model.state_dict(), checkpoint_file)
                patience_counter = 0
            else:
                if patience_counter < self.early_stopping_patience:
                    patience_counter += 1
                    self.logger.info("Validation result did not improve, patience counter is %s", patience_counter)
                else:
                    self.logger.info("Stopping training since model does not improve")
                    break
        if self.early_stopping and os.path.exists(checkpoint_file):
            rcnn_model.load_state_dict(torch.load(checkpoint_file))
            # Cleanup
            os.remove(checkpoint_file)

    def set_tunable_parameters(self, **kwargs):
        self.logger.warning("No parameters to be set..")
