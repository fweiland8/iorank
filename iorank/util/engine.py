import os

import logging
import torch
from torch.utils.data import DataLoader, Subset

from iorank.datasets.util import get_split
from iorank.util.util import get_uuid, get_root_dataset

"""
Engine providing utility methods for training and evaluation (PyTorch) models.

"""

logger = logging.getLogger("engine")


def evaluate(model, dataset, input_generator, monitor, batch_size=16):
    """
    Evaluates the given model.
    
    :param model: Model to evaluated
    :param dataset: Dataset on which the model is to evaluated
    :param input_generator: Input generator providing the input for the model
    :param monitor: Monitor to which the results are reported
    :param batch_size: Batch size. Default: 16
    :return: 
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    for i, data in enumerate(data_loader, 0):
        logger.debug("Start evaluation batch %s", i + 1)
        x, y = data
        x, kwargs, y = input_generator(x, y)
        with torch.no_grad():
            pred = model(x, **kwargs)
        monitor.add_val_result(pred, y)
        logger.debug("Finished evaluation batch %s", i + 1)


def train_one_epoch(model, dataloader, input_generator, loss_function, monitor, optimizer):
    """
    Performs a single training epoch.
    
    :param model: Model to trained
    :param dataloader: Data loader for the training set
    :param input_generator: Input generator providing the input for the model
    :param loss_function: Loss function to be used
    :param monitor: Monitor to which the results are reported
    :param optimizer: Optimizer for training
    """

    for i, data in enumerate(dataloader, 0):
        logger.debug("Start training batch %s", i + 1)
        x, y = data
        x, kwargs, y = input_generator(x, y)
        pred = model(x, **kwargs, targets=y)
        loss = loss_function(pred, y)
        loss_value = float(loss.mean().data)
        optimizer.zero_grad()
        logger.debug("Doing gradient step")
        loss.mean().backward()
        optimizer.step()
        monitor.add_train_result(pred, y, loss_value)
        logger.debug("Finished training batch %s", i + 1)


def train(model, dataset, input_generator, loss_function, monitor, optimizer, max_n_epochs=20, batch_size=16,
          early_stopping=True, early_stopping_patience=2, scheduler=None):
    """
    Method for training a model on the provided dataset.

    :param model: Model to trained
    :param dataset: Dataset for training/validation
    :param input_generator: Input generator providing the input for the model
    :param loss_function: Loss function to be used
    :param monitor: Monitor object to be used for training
    :param optimizer: Optimizer object
    :param max_n_epochs: Maximum number of training epochs. Default: 20
    :param batch_size: Batch size for training. Default: 16
    :param early_stopping: If True, early stopping is done. Default: True
    :param early_stopping_patience: Number of epoch without improvement before training is stopped. Default: 2
    :param scheduler: Learning rate scheduler. If no scheduler is provided, the learning rate
        is not scheduled.
    """
    dataset_full_size = len(dataset)
    logger.info("Start batch learning on dataset with %s instances", dataset_full_size)
    logger.info("Train up to %s epochs", max_n_epochs)
    val_split = 0.2

    split = get_split(dataset)
    train_indices, val_indices = split(dataset, val_split, shuffle=True)
    root_dataset = get_root_dataset(dataset)
    dataset_train = Subset(root_dataset, train_indices)
    dataset_val = Subset(root_dataset, val_indices)

    logger.info("Training on %s instances, validating on %s instances", len(dataset_train), len(dataset_val))
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, pin_memory=True)

    best_val_result = None

    checkpoint_file = "checkpoint_{}.pt".format(get_uuid())
    logger.debug("Write checkpoints to %s", checkpoint_file)

    patience_counter = 0
    for t in range(max_n_epochs):
        monitor.reset()
        logger.debug("Start epoch %s", t + 1)
        model.train()
        train_one_epoch(model, data_loader_train, input_generator, loss_function, monitor, optimizer)
        logger.info("[Epoch %s/%s]: training result: %s", t + 1, max_n_epochs, monitor.get_train_summary())
        model.eval()
        evaluate(model, dataset_val, input_generator, monitor, batch_size=batch_size)
        logger.info("[Epoch %s/%s]: validation result: %s", t + 1, max_n_epochs, monitor.get_val_summary())

        if scheduler is not None:
            scheduler.step()

        if not early_stopping:
            logger.info("Early stopping disabled, finishing epoch")
            continue

        if monitor.is_better_than(best_val_result):
            logger.info("Validation result is new best result, saving model..")
            best_val_result = monitor.get_val_score()
            save_checkpoint(model, checkpoint_file)
            patience_counter = 0
        else:
            if patience_counter < early_stopping_patience:
                patience_counter += 1
                logger.info("Validation result did not improve, patience counter is %s", patience_counter)
            else:
                logger.info("Stopping training since model does not improve")
                break
        monitor.reset()
    if early_stopping and os.path.exists(checkpoint_file):
        load_checkpoint(model, checkpoint_file)
        # Cleanup
        os.remove(checkpoint_file)


def save_checkpoint(model, checkpoint_file):
    """
    Creates a checkpoint for the provided model's state at the given file location.

    :param model: Model for which a checkpoint has to be created
    :param checkpoint_file: Filename of the checkpoint file
    :return:
    """
    torch_model = model.model
    torch.save(torch_model.state_dict(), checkpoint_file)


def load_checkpoint(model, checkpoint_file):
    """
    Loads the state for the given model from the provided checkpoint file

    :param model: Model which has to be loaded
    :param checkpoint_file: Filename of the checkpoint file
    """
    torch_model = model.model
    torch_model.load_state_dict(torch.load(checkpoint_file))
