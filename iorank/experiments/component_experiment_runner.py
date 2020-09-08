import os
import traceback

import argparse
import logging
import time
import yaml

from iorank.experiments.db_connector import DbConnector
from iorank.featuretransformation.combine_feature_transformer import CombineFeatureTransformer
from iorank.image_object_ranker_component import ImageObjectRankerComponent
from iorank.training.image_object_ranker_trainer import ImageObjectRankerTrainer
from iorank.training.tuning import HyperparameterTuner
from iorank.util.constants import object_rankers, feature_transformers, datasets, object_detectors
from iorank.util.util import get_hostname, setup_logging, read_ranges, string_to_kwargs

logger = logging.getLogger("experiment_runner")

"""
Script for running an experiment with the component model.
"""


def create_combine_transformer(configuration, dataset):
    """
    Creates a combiner feature transformer from the given configuration

    :param configuration: Experiment configuration
    :param dataset: Experiment dataset
    :return: The combiner transformer
    """
    n_classes = dataset.get_n_classes()

    # Read names and parameters of the individual feature transformers
    transformer_string = configuration["feature_transformer"]
    transformer_names = transformer_string.split(",")
    transformer_params_str = configuration["feature_transformer_params"]

    # No hyperparameters defined
    if transformer_params_str is None:
        transformers = [feature_transformers[name](n_classes=n_classes) for name in transformer_names]
    else:
        transformer_params = transformer_params_str.split("|")
        if len(transformer_names) != len(transformer_params):
            raise RuntimeError("Invalid configuration! {} transformers but {} params!".format(len(transformer_names),
                                                                                              len(transformer_params)))
        transformers = []
        # Turn parameter strings into kwargs dicts
        for name, params in zip(transformer_names, transformer_params):
            param_dict = {}
            if params:
                param_dict = string_to_kwargs(params)

            transformers.append(feature_transformers[name](n_classes=n_classes, **param_dict))
    transformer = CombineFeatureTransformer(transformers)
    return transformer


def create_ioranker(configuration, tool_config, dataset):
    """
    Constructs a component model for the given configuration.

    :param configuration: The experiment configuration
    :param tool_config: The tool configuration
    :param dataset: The experiment dataset
    :return: An image object ranker using the component architecture
    """
    n_objects = dataset.padding_size
    n_classes = dataset.get_n_classes()

    # Parse general model parameters
    model_params_string = configuration["model_params"]
    model_params = {}
    if model_params_string is not None:
        model_params = string_to_kwargs(model_params_string)

    # Parse object detector parameters and create object detector
    detector_params_string = configuration["object_detector_params"]
    detector_params = {}
    if detector_params_string is not None:
        detector_params = string_to_kwargs(detector_params_string)
    if "model_file" in detector_params.keys():
        detector_params["model_file"] = os.path.join(tool_config["models_dir"], detector_params["model_file"])
    detector = object_detectors[configuration["object_detector"]](n_classes=n_classes,
                                                                  max_n_objects=dataset.padding_size, **detector_params)

    # Parse feature transformer parameters and create feature transformer
    transformer_string = configuration["feature_transformer"]
    if "," in transformer_string:
        transformer = create_combine_transformer(configuration, dataset)
    else:
        transformer_params_string = configuration["feature_transformer_params"]
        transformer_params = {}
        if transformer_params_string is not None:
            transformer_params = string_to_kwargs(transformer_params_string)
        if "model_file" in transformer_params.keys():
            transformer_params["model_file"] = os.path.join(tool_config["models_dir"], transformer_params["model_file"])
        transformer = feature_transformers[configuration["feature_transformer"]](n_classes=n_classes,
                                                                                 **transformer_params)
    # Parse object ranker parameters and create object ranker
    ranker_params_string = configuration["object_ranker_params"]
    ranker_params = {}
    if ranker_params_string is not None:
        ranker_params = string_to_kwargs(ranker_params_string)
    if "model_file" in ranker_params.keys():
        ranker_params["model_file"] = os.path.join(tool_config["models_dir"], ranker_params["model_file"])
    n_object_features = transformer.get_n_features()
    ranker = object_rankers[configuration["object_ranker"]](n_object_features=n_object_features,
                                                            n_objects=n_objects, **ranker_params)

    ioranker = ImageObjectRankerComponent(detector, transformer, ranker, **model_params)
    return ioranker


def create_ioranker_trainer(configuration):
    """
    Creates a trainer object for training the component model.

    :param configuration: The experiment configuration
    :return: An image object ranker trainer
    """
    trainer = ImageObjectRankerTrainer()

    # Box expansion needs to be propagated to trainer
    model_params_string = configuration["model_params"]
    model_params = {}
    if model_params_string is not None:
        model_params = string_to_kwargs(model_params_string)
    if "box_expansion_factor" in model_params.keys():
        trainer.object_ranker_trainer_params["box_expansion_factor"] = model_params["box_expansion_factor"]

    return trainer


def run_component_model_experiment(configuration, tool_config, result_id):
    """
    Runs an experiment with the given experiment configuration and tool configuration.

    :param configuration: Experiment configuration
    :param tool_config: Tool configuration
    :param result_id: Id of the result entry in the database
    :return: dict containing the experiment results
    """

    # Setup logging
    log_filename = "iorank_conf_{}_result_{}.log".format(configuration["id"], result_id)
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
    augmentation = configuration.get("augmentation", False)
    dataset_train = datasets[configuration["dataset"]](root, mode='train', in_memory=in_memory,
                                                       augmentation=augmentation)
    dataset_test = datasets[configuration["dataset"]](root, mode='test', in_memory=in_memory)

    # Create ranker and trainer
    ioranker = create_ioranker(configuration, tool_config, dataset_train)
    trainer = create_ioranker_trainer(configuration)

    # Do training
    tuning_iterations = configuration["tuning_iterations"]
    if tuning_iterations == 0:
        logger.info("Skip hyperparameter optimization")
        trainer.prepare(ioranker)
        trainer.train(ioranker, dataset_train)
    else:
        logger.info("Do hyperparameter optimization with %s iterations", tuning_iterations)

        # Read parameter ranges for the model components
        model_parameter_ranges = {
            "object_detector": read_ranges(configuration["object_detector_param_ranges"]),
            "feature_transformer": read_ranges(configuration["feature_transformer_param_ranges"]),
            "object_ranker": read_ranges(configuration["object_ranker_param_ranges"])
        }

        # Read parameter ranges for the component trainers
        trainer_parameter_ranges = {
            "object_detector": read_ranges(configuration["object_detector_trainer_param_ranges"]),
            "feature_transformer": read_ranges(configuration["feature_transformer_trainer_param_ranges"]),
            "object_ranker": read_ranges(configuration["object_ranker_trainer_param_ranges"])
        }

        # Time limit must be known
        time_limit = os.environ.get('IORANK_TIME_LIMIT')
        if time_limit is None:
            raise RuntimeError("No time limit set for tuning")
        else:
            time_limit = int(time_limit)
        tuner = HyperparameterTuner(trainer, ioranker, model_parameter_ranges, trainer_parameter_ranges, time_limit,
                                    n_iterations=tuning_iterations)
        tuner.tune(dataset_train)

    logger.info("Doing evaluation..")
    result = trainer.evaluate(ioranker, dataset_test)
    logger.info("Finished evaluation")
    return result


def run_next_experiment(tool_config):
    """
    Fetches an experiment from the database and runs that experiment.

    :param tool_config: Tool configuration
    :return: Experiment result
    """

    db_config = tool_config["db"]
    dbcon = DbConnector(db_config["user"], db_config["password"], db_config["host"], db_config["database"],
                        db_config["schema"])
    host = get_hostname()
    experiment_configuration, result_id = dbcon.get_next_experiment(host)
    if experiment_configuration is None:
        print("No experiments left to be done")
        return
    time_start = time.time()
    try:
        result = run_component_model_experiment(experiment_configuration, tool_config, result_id)
        duration = int(time.time() - time_start)
        result["duration"] = duration
        result["result_id"] = result_id
        logger.info("Experiment finished successfully, result is: %s", result)
        dbcon.set_result_success(result)
    except Exception:
        logger.exception("Experiment for configuration %s failed:", experiment_configuration["id"])
        exception_str = traceback.format_exc()
        result = {"exception": exception_str}
        duration = int(time.time() - time_start)
        result["duration"] = duration
        result["result_id"] = result_id
        logger.info("Experiment finished with errors, result is: %s", result)
        dbcon.set_result_exception(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration file', type=str, required=True)
    args = parser.parse_args()
    config_file = args.config
    if not os.path.isfile(config_file):
        raise RuntimeError("Config file must be provided")
    tool_config = yaml.safe_load(open(config_file))
    run_next_experiment(tool_config)
