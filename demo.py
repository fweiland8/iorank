import os

import argparse
import pathlib
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from iorank.featuretransformation.deep_feature_transformer import ResNetFeatureTransformer
from iorank.image_object_ranker_component import ImageObjectRankerComponent
from iorank.objectdetection.faster_rcnn import FasterRCNN
from iorank.objectranking.torch_fate_ranker import TorchFATERanker
from iorank.util.util import get_device, scores_to_rankings_with_size_torch
from iorank.visualization.visualization import show_image_with_boxes

"""
Small demo application, which instantiates a component model in order to make prediction for a given image. 

"""


def create_ioranker():
    # Use a pretrained (not-finetuned) Faster R-CNN model for the demo
    detector = FasterRCNN(1, finetuning=False, confidence_threshold=0.7)
    transformer = ResNetFeatureTransformer(reduced_size=128)
    ranker = TorchFATERanker(n_object_features=transformer.get_n_features())

    # Load pretrained weights for the FATE ranker
    current_dir = pathlib.Path(__file__).parent.absolute()
    file = os.path.join(current_dir, "models", "fate_pretrained_resnet.pt")
    if not os.path.exists(file):
        raise RuntimeError("Missing file: {}".format(file))
    ranker.model.load_state_dict(torch.load(file, map_location=get_device()))
    ranker.eval()
    ioranker = ImageObjectRankerComponent(detector, transformer, ranker)

    return ioranker


parser = argparse.ArgumentParser(description='Image Object Ranking demo. Instantiates a model with pretrained weights '
                                             'in order to process the given input image.')
parser.add_argument('-i', '--input', help='Path to the input image', required=True)
args = vars(parser.parse_args())

ioranker = create_ioranker()

input_file = args["input"]
if not os.path.exists(input_file):
    raise RuntimeError("Input file {} does not exist".format(input_file))

image = Image.open(input_file).convert("RGB")
image = TF.to_tensor(image)

image_t = image.unsqueeze(0)

prediction = ioranker(image_t)

# Visualize results
boxes = prediction["boxes"][0]
scores = prediction["scores"]
ranking_sizes = torch.sum(torch.ne(prediction["conf"], -1), dim=1)
ranking = scores_to_rankings_with_size_torch(scores, ranking_sizes)[0]
show_image_with_boxes(image, boxes, ranking=ranking)

print("Done")
