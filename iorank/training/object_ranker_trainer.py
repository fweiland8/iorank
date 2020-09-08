import logging
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from iorank.metrics.metrics import kendalls_tau
from iorank.training.monitor import DefaultMonitor
from iorank.util.engine import train
from iorank.util.names import losses
from iorank.util.util import scores_to_rankings_torch, get_device, expand_boxes


def create_input_generator(feature_transformer, box_expansion_factor):
    """
    Creates an input generator for training an object ranker.

    :param feature_transformer: Feature transformer producing the feature vectors for training
    :param box_expansion_factor: Factor by which bounding boxes can be expanded in order to
    add additional context information
    :return: The input generator function
    """

    def input_generator(x, y):
        boxes = y["boxes"]
        labels = y["labels"]
        images = x
        images = images.to(get_device(), non_blocking=True)

        masks = None
        if "masks" in y.keys():
            masks = y["masks"]

        # Expand bounding boxes
        if box_expansion_factor != 0:
            image_height, image_width = images.size()[-2:]
            boxes = expand_boxes(boxes, box_expansion_factor,
                                 image_height,
                                 image_width)

        # Create feature vectors
        feature_vectors = feature_transformer.transform(images, boxes, labels, masks)
        feature_vectors = feature_vectors.to(get_device(), non_blocking=True)
        return feature_vectors, {"images": images}, y

    return input_generator


class ObjectRankerMonitor(DefaultMonitor):
    def add_train_result(self, pred, y, loss_value):
        # Wrap the predicted scores into a dict in order to use the standard metric methods
        pred = {"scores": pred}
        super().add_train_result(pred, y, loss_value)

    def add_val_result(self, pred, y):
        # Wrap the predicted scores into a dict in order to use the standard metric methods
        pred = {"scores": pred}
        super().add_val_result(pred, y)


class ObjectRankerTrainer:
    def __init__(self, loss="hinged_rank_loss", max_n_epochs=40,
                 optimizer="Adam", lr=1e-4, batch_size=32, box_expansion_factor=0, **kwargs):
        """
        Creates a trainer for training object ranking models.

        :param loss: Loss function to be used. Is only considered for PyTorch models. Default: 'hinged_rank_loss'
        :param max_n_epochs: Maximum number of training epochs. Default: 40
        :param optimizer: Optimizer to be used for training. Default: Adam
        :param lr: Initial learning rate. Default: 1e-4
        :param batch_size: Batch size for training. Is only considered for PyTorch models. Default: 16
        :param box_expansion_factor: Factor by which bounding boxes can be expanded in order to
        add additional context information. Default: 0 (no expansion)
        :param kwargs: Keyword arguments
        """

        # Metrics for validation
        self.metrics = {"kendalls_tau": kendalls_tau}
        self.loss = loss
        self.max_n_epochs = max_n_epochs
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.box_expansion_factor = box_expansion_factor
        self.logger = logging.getLogger(ObjectRankerTrainer.__name__)

    def train(self, object_ranker, feature_transformer, dataset):
        """
        Trains the given object ranker on the provided dataset. For producing the required
        feature vectors, the provided feature transformer is used.

        :param object_ranker: Object ranker to be trained
        :param feature_transformer: Feature transformer model
        :param dataset: Test dataset
        """

        # Training is dependent on the implementation
        if object_ranker.torch_model:
            self._train_torch(object_ranker, feature_transformer, dataset)
        else:
            self._train_default(object_ranker, feature_transformer, dataset)

    def _train_torch(self, object_ranker, feature_transformer, dataset):
        """
        Trains the given PyTorch object ranker. In PyTorch training, the feature vectors are generated for each batch.

        :param object_ranker: Object ranker to be trained
        :param feature_transformer: Feature transformer model
        :param dataset: Test dataset
        """
        self.logger.info("Start training torch model with optimizer=%s, lr=%s, batch_size=%s, max_n_epochs=%s",
                         self.optimizer, self.lr, self.batch_size, self.max_n_epochs)

        opt, scheduler = self._create_optimizer_scheduler(object_ranker.parameters())
        loss_function = losses[self.loss]

        # Build wrapper method for the loss as the scores need to be turned into a ranking first
        def loss(s_pred, y):
            s_true = y["scores"]
            ranking_true = scores_to_rankings_torch(s_true)
            l = loss_function(s_pred, ranking_true)
            return l

        monitor = ObjectRankerMonitor(self.metrics, loss_function)

        # Ensure that model is reset before training
        object_ranker.reset()

        # Do training
        train(object_ranker, dataset, create_input_generator(feature_transformer, self.box_expansion_factor), loss,
              monitor, opt,
              max_n_epochs=self.max_n_epochs, early_stopping_patience=3, batch_size=self.batch_size,
              scheduler=scheduler)

    def _train_default(self, object_ranker, feature_transformer, dataset):
        """
        Trains a non PyTorch object ranker. First, feature vectors are produced for the entire dataset, then the
        object ranker is trained using these feature vectors.

        :param object_ranker: Object ranker to be trained
        :param feature_transformer: Feature transformer model
        :param dataset: Test dataset
        """
        self.logger.info("Start training on entire dataset")
        # Train on entire dataset
        data_loader = DataLoader(dataset, batch_size=16, pin_memory=True)

        self.logger.info("Start feature transformation")
        features = None
        scores = None
        for i, data in enumerate(data_loader, 0):
            self.logger.debug("Batch %s", i)
            images, gt = data
            boxes = gt["boxes"]
            labels = gt["labels"]
            f = feature_transformer.transform(images, boxes, labels)
            if not torch.is_tensor(f):
                f = torch.tensor(f)
            s = scores_to_rankings_torch(gt["scores"])

            # CUDA is not suitable for non-torch models
            f = f.detach().cpu()
            s = s.cpu()

            if features is None:
                features = f
            else:
                features = torch.cat((features, f))

            if scores is None:
                scores = s
            else:
                scores = torch.cat((scores, s))
        self.logger.info("Finished feature transformation")
        self.logger.info("Start training ranker")
        object_ranker.fit(features, scores, verbose=True, epochs=self.max_n_epochs)
        self.logger.info("Finished training ranker")

    def set_tunable_parameters(self, loss="hinged_rank_loss", max_n_epochs=40, optimizer="Adam", lr=1e-4,
                               batch_size=32):
        """
        Sets the tunable parameters for this trainer.

        :param loss: Loss function to be used. Is only considered for PyTorch models. Default: 'hinged_rank_loss'
        :param max_n_epochs: Maximum number of training epochs. Default: 40
        :param optimizer: Optimizer to be used for training. Default: Adam
        :param lr: Initial learning rate. Default: 1e-4
        :param batch_size: Batch size for training. Is only considered for PyTorch models. Default: 16
        """
        self.loss = loss
        self.max_n_epochs = max_n_epochs
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size

    def _create_optimizer_scheduler(self, parameters):
        """
        Creates an optimizer and learning rate scheduler for training.

        :param parameters: Model parameters of the object ranker
        :return:
        """
        optimizer_name = self.optimizer
        if optimizer_name == "Adam":
            opt = Adam(parameters, lr=self.lr, weight_decay=1e-5)
            scheduler = None
        elif optimizer_name == "SGD":
            opt = SGD(parameters, lr=self.lr, nesterov=True, momentum=0.9, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 3, gamma=0.8)
        else:
            raise RuntimeError("Invalid optimizer name: {}".format(optimizer_name))
        return opt, scheduler
