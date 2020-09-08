from iorank.datasets.kitti_dataset import KittiDataset
from iorank.datasets.kitti_mask_dataset import KittiMaskDataset
from iorank.featuretransformation.autoencoder_conv import AutoEncoderFeatureTransformer
from iorank.featuretransformation.bounding_box_feature_transformer import BoundingBoxFeatureTransformer
from iorank.featuretransformation.deep_feature_transformer import AlexNetFeatureTransformer, ResNetFeatureTransformer
from iorank.featuretransformation.handcrafted_feature_transformer import HandcraftedFeatureTransformer
from iorank.featuretransformation.label_feature_transformer import LabelFeatureTransformer
from iorank.featuretransformation.raw_feature_transformer import RawFeatureTransformer
from iorank.featuretransformation.spatial_mask_feature_transformer import SpatialMaskFeatureTransformer
from iorank.metrics.metrics import reduced_ranking, \
    kendalls_tau, spearman, zero_one_accuracy, object_detection_precision, \
    object_detection_recall, average_ranking_size, label_accuracy
from iorank.objectdetection.faster_rcnn import FasterRCNN
from iorank.objectdetection.mask_rcnn import MaskRCNN
from iorank.objectdetection.yolo import YOLOObjectDetector
from iorank.objectranking.baseline_ranker import BaselineRanker
from iorank.objectranking.cmp_net_ranker import CmpNetRanker
from iorank.objectranking.fate_ranker import FATERanker
from iorank.objectranking.feta_ranker import FETARanker
from iorank.objectranking.random_ranker import RandomRanker
from iorank.objectranking.rank_net_ranker import RankNetRanker
from iorank.objectranking.rank_svm_ranker import RankSVMRanker
from iorank.objectranking.torch_fate_ranker import TorchFATERanker
from iorank.objectranking.torch_feta_ranker import TorchFETARanker

datasets = {"KITTI": KittiDataset,
            "MaskKITTI": KittiMaskDataset}

object_detectors = {"faster-rcnn": FasterRCNN,
                    "mask-rcnn": MaskRCNN,
                    "yolo": YOLOObjectDetector}

feature_transformers = {"bounding-box": BoundingBoxFeatureTransformer,
                        "autoencoder": AutoEncoderFeatureTransformer,
                        "alexnet": AlexNetFeatureTransformer,
                        "resnet": ResNetFeatureTransformer,
                        "raw": RawFeatureTransformer,
                        "spatial-mask": SpatialMaskFeatureTransformer,
                        "label": LabelFeatureTransformer,
                        "handcrafted": HandcraftedFeatureTransformer}

object_rankers = {"fate-ranker": FATERanker,
                  "feta-ranker": FETARanker,
                  "torch-fate-ranker": TorchFATERanker,
                  "torch-feta-ranker": TorchFETARanker,
                  "baseline-ranker": BaselineRanker,
                  "random-ranker": RandomRanker,
                  "rank-svm-ranker": RankSVMRanker,
                  "rank-net-ranker": RankNetRanker,
                  "cmp-net-ranker": CmpNetRanker}

metrics = {"kendalls_tau": reduced_ranking(kendalls_tau),
           "spearman": reduced_ranking(spearman),
           "zero_one_accuracy": reduced_ranking(zero_one_accuracy),
           "object_detection_precision": object_detection_precision,
           "object_detection_recall": object_detection_recall,
           "avg_ranking_size": reduced_ranking(average_ranking_size),
           "label_accuracy": reduced_ranking(label_accuracy)}
