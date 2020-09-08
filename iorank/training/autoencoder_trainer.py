import random

import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD

from iorank.util.engine import train
from iorank.util.util import get_patch_from_image, get_device, is_dummy_box

logger = logging.getLogger("autoencoder_training")


def get_random_patches(rgb_images, n_patches=16, height_interval=(50, 100), width_interval=(100, 200)):
    """
    Extracts random patches from the given images. The size as well as the position of the patches is randomized.

    :param rgb_images: Images from which patches are to be extracted
    :param n_patches: Number of patches per image
    :param height_interval: Interval from which the patch height is selected at random
    :param width_interval: Interval from which the patch width is selected at random
    :return: List of patches
    """
    patches = []
    for rgb_image in rgb_images:
        image_height = rgb_image.size()[1]
        image_width = rgb_image.size()[2]
        for _ in range(n_patches):
            patch_height = random.randrange(height_interval[0], height_interval[1])
            patch_width = random.randrange(width_interval[0], width_interval[1])
            x0 = random.randrange(0, image_width - patch_width)
            y0 = random.randrange(0, image_height - patch_height)
            patch = [x0, y0, x0 + patch_width, y0 + patch_height]
            patches.append(get_patch_from_image(patch, rgb_image))
    return patches


def get_object_patches(rgb_images, all_boxes):
    """
    Extracts object patches from the given images.

    N: Batch size \n
    U: Upper bound for the number of objects (padding size)

    :param rgb_images: Images from which patches are to be extracted
    :param all_boxes: Bounding box coordinates. Tensor of size (N,U,4)
    :return: List of patches
    """
    patches = []
    for rgb_image, boxes in zip(rgb_images, all_boxes):
        for box in boxes:
            if is_dummy_box(box):
                continue
            patches.append(get_patch_from_image(box, rgb_image))
    return patches


def resize_patches(patches, size):
    """
    Resizes the given patches to a uniform size.

    :param patches: Patches to be resized
    :param size: Size (side length) to which the patches are resized
    :return: A tensor of patches
    """
    ret = []
    for patch in patches:
        patch = patch.expand(1, -1, -1, -1)
        ret.append(F.interpolate(patch, size=(int(size), int(size))))
    return torch.cat(ret)


class AutoEncoderMonitor:
    def __init__(self, loss):
        """
        Creates a monitor instance for the auto encoder training.

        :type loss: function
        :param loss: The loss function used in training
        """
        self.loss = loss
        self.train_losses = []
        self.val_losses = []

    def add_train_result(self, pred, y, loss_value):
        """
        This method is called for each training batch.

        :param pred: Predicted data
        :param y: Ground truth data
        :param loss_value: Loss value from the loss function
        """

        self.train_losses.append(loss_value)

    def add_val_result(self, pred, y):
        """
        This method is called for each validation batch.

        :param pred: Predicted data
        :param y: Ground truth data
        """

        self.val_losses.append(self.loss(pred, y).cpu().numpy())

    def get_train_summary(self):
        """
        Returns a summary of the achieved training results.

        :return: A string with the average loss in training
        """
        return "loss = {:0.5f}".format(np.mean(self.train_losses))

    def get_val_summary(self):
        """
        Returns a summary of the achieved validation results.

        :return: A string with the average loss in validation
        """
        return "loss = {:0.5f}".format(np.mean(self.val_losses))

    def get_val_score(self):
        """
        Returns a score describing the performance on the validation set.

        :return: Validation score
        """
        return np.mean(self.val_losses)

    def reset(self):
        """
        This method is called after each epoch in order to reset the results.

        """
        self.train_losses = []
        self.val_losses = []

    def is_better_than(self, best_val_result):
        """
        Compares the provided result (from a preceding epoch) to the achieved results in the current epoch.

        :param best_val_result: Validation result to which the current results are to be compared
        :return: True, if the current result is better than the provided result, False otherwise
        """
        if best_val_result is None:
            return True
        val_score = self.get_val_score()
        # Simply compare the scores
        # As a loss is considered, lower is better
        if val_score < best_val_result:
            return True
        else:
            return False


def create_input_generator_object_patches(reduced_size):
    """
    Creates the input generator function for auto encoder training with object patches.

    :param reduced_size: Size to which the input patches are resized
    :return: Input generator function
    """

    def input_generator(x, y):
        boxes = y["boxes"]
        patches = get_object_patches(x, boxes)
        inputs = resize_patches(patches, reduced_size)
        return inputs.to(get_device()), {}, inputs

    return input_generator


def create_input_generator_random_patches(reduced_size):
    """
    Creates the input generator function for auto encoder training with random patches.

    :param reduced_size: Size to which the input patches are resized
    :return: Input generator function
    """

    def input_generator(x, y):
        patches = get_random_patches(x)
        inputs = resize_patches(patches, reduced_size)
        return inputs.to(get_device()), {}, inputs

    return input_generator


class AutoEncoderTrainer:
    def __init__(self, input_type='random_patches', optimizer="Adam", lr=1e-4, max_n_epochs=40):
        """
        Creates an Auto Encoder trainer.


        :param input_type: Input type for training. Either 'random_patches' or 'object_patches'.
        Default: 'random_patches'
        :param optimizer: Optimizer to be used for training. Default: Adam
        :param lr: Initial learning rate. Default: 1e-4
        :param max_n_epochs: Maximum number of training epochs. Default: 40
        """

        self.logger = logging.getLogger(AutoEncoderTrainer.__name__)
        self.max_n_epochs = max_n_epochs
        self.optimizer = optimizer

        # Tunable parameters
        self.lr = lr
        self.input_type = input_type

    def train(self, autoencoder, dataset):
        """
        Trains the provided auto encoder on the given dataset.

        :param autoencoder: Auto encoder to be trained
        :param dataset: Dataset on which the training takes place
        """

        # Define loss function (mse loss)
        def loss(pred, y):
            pred = pred.to(get_device())
            y = y.to(get_device())
            return F.mse_loss(pred, y)

        # Prepare training
        monitor = AutoEncoderMonitor(loss)
        opt, scheduler = self._create_optimizer_scheduler(autoencoder.parameters())
        loss_function = loss

        # Ensure that model is reset before training
        autoencoder.reset()

        # Create input generator
        if self.input_type == 'random_patches':
            input_generator = create_input_generator_random_patches(autoencoder.get_reduced_size())
        elif self.input_type == 'object_patches':
            input_generator = create_input_generator_object_patches(autoencoder.get_reduced_size())
        else:
            raise RuntimeError("Invalid autoencoder training input type: {}".format(self.input_type))
        train(autoencoder, dataset, input_generator, loss_function, monitor, opt, max_n_epochs=self.max_n_epochs,
              scheduler=scheduler, early_stopping_patience=4)

    def set_tunable_parameters(self, lr=1e-4, input_type='random_patches'):
        """
        Sets the tunable parameters for this trainer.

        :param lr: Initial learning rate. Default: 1e-4
        :param input_type: Input type for training. Either 'random_patches' or 'object_patches'.
        Default: 'random_patches'
        """
        self.lr = lr
        self.input_type = input_type

    def _create_optimizer_scheduler(self, parameters):
        """
        Creates an optimizer and learning rate scheduler for training.

        :param parameters: Model parameters of the object ranker
        :return:
        """
        optimizer_name = self.optimizer
        if optimizer_name == "Adam":
            opt = Adam(parameters, lr=self.lr, weight_decay=1e-5)
        elif optimizer_name == "SGD":
            opt = SGD(parameters, lr=self.lr, nesterov=True, momentum=0.9, weight_decay=1e-5)
        else:
            raise RuntimeError("Invalid optimizer name: {}".format(optimizer_name))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 3, gamma=0.9)
        return opt, scheduler
