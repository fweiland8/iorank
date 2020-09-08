import argparse

from iorank.datasets.kitti_dataset import KittiDataset
from iorank.image_object_ranker_e2e import ImageObjectRankerE2E
from iorank.training.image_object_ranker_e2e_trainer import ImageObjectRankerE2ETrainer
from iorank.util.util import setup_logging

"""
Small example, in which an E2E model is trained and evaluated.

"""

parser = argparse.ArgumentParser(description='Train an E2E model for Image Object Ranking.')
parser.add_argument('-d', '--dataset_dir', help='Directory to the dataset', required=True)
parser.add_argument('--backbone', help='Backbone to be used in the E2E Model',
                    default='resnet', required=False,
                    choices=["resnet", "mobilenet"])
parser.add_argument('--cells', help='Number of cells (in a row) for the YOLO object detector',
                    default=7, required=False, type=int, choices=[7, 11, 14])
args = vars(parser.parse_args())

setup_logging()

dataset_train = KittiDataset(root=args["dataset_dir"], mode='train')
dataset_test = KittiDataset(root=args["dataset_dir"], mode='test')
n_classes = dataset_train.get_n_classes()

ioranker = ImageObjectRankerE2E(n_classes=n_classes, max_n_objects=dataset_train.padding_size,
                                backbone_name=args["backbone"], n_cells=args["cells"])

trainer = ImageObjectRankerE2ETrainer()

ioranker.train()
trainer.train(ioranker, dataset_train)

ioranker.eval()
result = trainer.evaluate(ioranker, dataset_test)

print("Result:")
print("Precision (Object Detection): {}".format(result["object_detection_precision"]))
print("Recall (Object Detection): {}".format(result["object_detection_recall"]))
print("Label Accuracy: {}".format(result["label_accuracy"]))
print("Kendall's Tau: {}".format(result["kendalls_tau"]))
print("Spearman's Rho: {}".format(result["spearman"]))
print("0/1 ranking accuracy: {}".format(result["zero_one_accuracy"]))
