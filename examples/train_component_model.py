import argparse

from iorank.datasets.kitti_dataset import KittiDataset
from iorank.image_object_ranker_component import ImageObjectRankerComponent
from iorank.training.image_object_ranker_trainer import ImageObjectRankerTrainer
from iorank.util.constants import feature_transformers, object_detectors, object_rankers
from iorank.util.util import setup_logging

"""
Small example, in which a component model is trained and evaluated.

"""

parser = argparse.ArgumentParser(description='Train a component model for Image Object Ranking.')
parser.add_argument('-d', '--dataset_dir', help='Directory to the dataset', required=True)
parser.add_argument('--feature_transformer', help='Component realization for the feature transformer component',
                    default='resnet', required=False,
                    choices=feature_transformers.keys())
parser.add_argument('--object_detector', help='Component realization for the object detector component',
                    default='faster-rcnn', required=False,
                    choices=object_detectors.keys())
parser.add_argument('--object_ranker', help='Component realization for the object ranker component',
                    default='torch-fate-ranker', required=False,
                    choices=object_rankers.keys())
args = vars(parser.parse_args())

setup_logging()

dataset_train = KittiDataset(root=args["dataset_dir"], mode='train')
dataset_test = KittiDataset(root=args["dataset_dir"], mode='test')
n_classes = dataset_train.get_n_classes()

detector = object_detectors[args["object_detector"]](n_classes=n_classes)
transformer = feature_transformers[args["feature_transformer"]]()
ranker = object_rankers[args["object_ranker"]](n_object_features=transformer.get_n_features())
ioranker = ImageObjectRankerComponent(detector, transformer, ranker)

trainer = ImageObjectRankerTrainer()
trainer.prepare(ioranker)
trainer.train(ioranker, dataset_train)

result = trainer.evaluate(ioranker, dataset_test)

print("Result:")
print("Precision (Object Detection): {}".format(result["object_detection_precision"]))
print("Recall (Object Detection): {}".format(result["object_detection_recall"]))
print("Label Accuracy: {}".format(result["label_accuracy"]))
print("Kendall's Tau: {}".format(result["kendalls_tau"]))
print("Spearman's Rho: {}".format(result["spearman"]))
print("0/1 ranking accuracy: {}".format(result["zero_one_accuracy"]))
