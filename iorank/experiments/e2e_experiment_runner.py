import os
import traceback

import argparse
import logging
import time
import yaml

from iorank.experiments.db_connector import DbConnector
from iorank.image_object_ranker_e2e import ImageObjectRankerE2E
from iorank.training.image_object_ranker_e2e_trainer import ImageObjectRankerE2ETrainer
from iorank.util.constants import datasets
from iorank.util.util import get_hostname, setup_logging, get_device

logger = logging.getLogger("e2e_experiment_runner")

"""
Script for running an experiment with the E2E model.
"""


def create_e2e_ranker(configuration, dataset):
    """
    Creates a E2E ranker according to the given configuration and for the given dataset.

    :param configuration: Experiment configuration
    :param dataset: Dataset for which the model has to be constructed
    :return: An E2E ranker
    """
    n_cells = configuration["n_cells"]
    predictions_per_cell = configuration["predictions_per_cell"]
    n_classes = dataset.get_n_classes()
    input_size = configuration["input_size"]
    max_n_objects = dataset.padding_size
    backbone_name = configuration["backbone_name"]
    include_spatial_mask = configuration["include_spatial_mask"]
    feature_reduction = configuration.get("feature_reduction")

    ranker = ImageObjectRankerE2E(n_classes=n_classes, max_n_objects=max_n_objects, n_cells=n_cells,
                                  predictions_per_cell=predictions_per_cell,
                                  input_size=input_size, backbone_name=backbone_name,
                                  include_spatial_mask=include_spatial_mask,
                                  feature_reduction=feature_reduction)
    return ranker


def create_trainer(configuration):
    """
    Create trainer for the E2E ranker using the given configuration.

    :param configuration: Experiment configuration
    :return: Trainer for the E2E ranker.
    """
    od_loss_factor = configuration["od_loss_factor"]
    or_loss_factor = configuration["or_loss_factor"]
    max_n_epochs = configuration["max_n_epochs"]
    lr_decay_epoch = configuration["lr_decay_epoch"]
    batch_size = configuration["batch_size"]

    # Loss is a weighted sum of detection loss and ranking loss
    def loss(od_loss, r_loss):
        return od_loss_factor * od_loss + or_loss_factor * r_loss

    # Do not use lr decay
    if lr_decay_epoch == -1:
        scheduler = None
    else:
        def lr_scheduler(epoch):
            # Start lr decay after 'lr_decay_epoch' epochs
            if epoch < lr_decay_epoch:
                lr_factor = 1
            else:
                lr_factor = max(0.95 ** (epoch - lr_decay_epoch), 0.1)
            return lr_factor

        scheduler = lr_scheduler

    trainer = ImageObjectRankerE2ETrainer(loss_function=loss, batch_size=batch_size, lr_scheduler=scheduler,
                                          max_n_epochs=max_n_epochs, early_stopping=True)

    return trainer


def run_e2e_experiment(configuration, tool_config, result_id):
    """
    Runs an experiment with the given experiment configuration and tool configuration.

    :param configuration: Experiment configuration
    :param tool_config: Tool configuration
    :param result_id: Id of the result entry in the database
    :return: dict containing the experiment results
    """

    # Setup logging
    log_filename = "e2e_conf_{}_result_{}.log".format(configuration["id"], result_id)
    loglevel = os.environ.get('IORANK_LOGLEVEL')
    logfile_path = os.path.join(tool_config["logs_dir"], log_filename)
    if loglevel is not None:
        setup_logging(filename=logfile_path, loglevel=loglevel)
    else:
        setup_logging(filename=logfile_path)
    logger.info("Starting experiment with configuration %s", configuration)

    # Load dataset
    root = tool_config["dataset_root"]
    in_memory = tool_config.get("dataset_preloading", False)
    dataset_train = datasets[configuration["dataset"]](root, mode='train', in_memory=in_memory, augmentation=True)
    dataset_test = datasets[configuration["dataset"]](root, mode='test', in_memory=in_memory)

    # Create ranker and trainer
    ranker = create_e2e_ranker(configuration, dataset_train)
    ranker.to(get_device())
    ranker.train()
    trainer = create_trainer(configuration)

    # Do training
    logger.info("Start training..")
    trainer.train(ranker, dataset_train)
    logger.info("Finished training")

    ranker.eval()

    # Do evaluation
    logger.info("Doing evaluation..")
    result = trainer.evaluate(ranker, dataset_test)
    logger.info("Finished evaluation")
    return result


def run_next_experiment(tool_config):
    """
    Fetches an experiment from the database and runs that experiment.

    :param tool_config: Tool configuration
    :return: Experiment result
    """

    # Try to fetch next experiment from database
    db_config = tool_config["db"]
    dbcon = DbConnector(db_config["user"], db_config["password"], db_config["host"], db_config["database"],
                        db_config["schema"])
    host = get_hostname()
    experiment_configuration, result_id = dbcon.get_next_experiment(host, e2e=True)

    # No more experiments
    if experiment_configuration is None:
        print("No experiments left to be done")
        return
    time_start = time.time()
    try:
        result = run_e2e_experiment(experiment_configuration, tool_config, result_id)
        duration = int(time.time() - time_start)
        result["duration"] = duration
        result["result_id"] = result_id
        logger.info("Experiment finished successfully, result is: %s", result)
        dbcon.set_result_success(result, e2e=True)
    except Exception:
        logger.exception("Experiment for configuration %s failed:", experiment_configuration["id"])
        exception_str = traceback.format_exc()
        result = {"exception": exception_str}
        duration = int(time.time() - time_start)
        result["duration"] = duration
        result["result_id"] = result_id
        logger.info("Experiment finished with errors, result is: %s", result)
        dbcon.set_result_exception(result, e2e=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration file', type=str, required=True)
    args = parser.parse_args()
    config_file = args.config
    if not os.path.isfile(config_file):
        raise RuntimeError("Config file must be provided")
    tool_config = yaml.safe_load(open(config_file))
    run_next_experiment(tool_config)
