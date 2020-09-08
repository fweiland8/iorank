from torch import nn
from torchvision import models


class MobileNetBackbone(nn.Module):

    def __init__(self, input_size, pretrained=True):
        """
        Creates an instance of the MobileNetV2 backbone. The MobileNetV2 network is reduced to its feature extraction
        part, i.e. the fully-connected layers for classification are removed.

        :param input_size: Side length of the images that are processed in this network
        :param pretrained: If True, pretrained weights are used. Default: True
        """
        super(MobileNetBackbone, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.model = list(mobilenet.children())[0]
        self.conv_map_size = input_size / 32
        self.conv_map_depth = 1280
        self.n_features = self.conv_map_depth * self.conv_map_size ** 2

    def forward(self, x):
        return self.model(x)


class ResNetBackbone(nn.Module):
    def __init__(self, input_size, pretrained=True):
        """
        Creates an instance of the ResNet backbone. The network is reduced to its feature extraction
        part, i.e. the fully-connected layers for classification are removed.

        :param input_size: Side length of the images that are processed in this network
        :param pretrained: If True, pretrained weights are used. Default: True
        """
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.model = nn.Sequential(*list(resnet.children())[:-2])
        self.conv_map_size = input_size / 32
        self.conv_map_depth = 2048
        self.n_features = self.conv_map_depth * self.conv_map_size ** 2

    def forward(self, x):
        return self.model(x)
