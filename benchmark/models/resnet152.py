import torch.nn as nn
from . import ImageClassifier

## ResNet-152 Model


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, indentity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.indentity_downsample = indentity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.indentity_downsample is not None:
            identity = self.indentity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


"""
  here we represent the Resnet which start with non-residual layers as follows:
  a Conv with kernel size of 7 x 7 ---> Batch normlization ---> ReLU Function
  ---> Maxpooling
  after that we present the residual layers with 4 blocks each block repeated
  (3, 8, 36, 3) respectively. the block architecture implemented in the Block class above.
"""


class ResNet(ImageClassifier):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        indentity_downsample = None
        layers = []

        # check for the identitiy layer so we know when to add a skip connection
        if stride != 1 or self.in_channels != out_channels * 4:
            indentity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * 4, kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(out_channels * 4),
            )
        layers.append(
            block(self.in_channels, out_channels, indentity_downsample, stride)
        )
        self.in_channels = out_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)


def ResNet152(image_channels, num_classes):
    return ResNet(
        Block, [3, 8, 36, 3], image_channels=image_channels, num_classes=num_classes
    )
