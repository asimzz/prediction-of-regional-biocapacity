import torch
import torch.nn as nn
from . import ImageClassifier


class GoogleNet(ImageClassifier):
    def __init__(self, in_channels, num_classes):
        super(GoogleNet, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(
            in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(
            in_channels=192,
            out_1x1=64,
            red_3x3=96,
            out_3x3=128,
            red_5x5=16,
            out_5x5=32,
            out_1x1pool=32,
        )
        self.inception3b = InceptionBlock(
            in_channels=256,
            out_1x1=128,
            red_3x3=128,
            out_3x3=192,
            red_5x5=32,
            out_5x5=96,
            out_1x1pool=64,
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionBlock(
            in_channels=480,
            out_1x1=192,
            red_3x3=96,
            out_3x3=208,
            red_5x5=16,
            out_5x5=48,
            out_1x1pool=64,
        )
        self.inception4b = InceptionBlock(
            in_channels=512,
            out_1x1=160,
            red_3x3=112,
            out_3x3=224,
            red_5x5=24,
            out_5x5=64,
            out_1x1pool=64,
        )
        self.inception4c = InceptionBlock(
            in_channels=512,
            out_1x1=128,
            red_3x3=128,
            out_3x3=256,
            red_5x5=24,
            out_5x5=64,
            out_1x1pool=64,
        )
        self.inception4d = InceptionBlock(
            in_channels=512,
            out_1x1=112,
            red_3x3=144,
            out_3x3=288,
            red_5x5=32,
            out_5x5=64,
            out_1x1pool=64,
        )
        self.inception4e = InceptionBlock(
            in_channels=528,
            out_1x1=256,
            red_3x3=160,
            out_3x3=320,
            red_5x5=32,
            out_5x5=128,
            out_1x1pool=128,
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(
            in_channels=832,
            out_1x1=256,
            red_3x3=160,
            out_3x3=320,
            red_5x5=32,
            out_5x5=128,
            out_1x1pool=128,
        )
        self.inception5b = InceptionBlock(
            in_channels=832,
            out_1x1=384,
            red_3x3=192,
            out_3x3=384,
            red_5x5=48,
            out_5x5=128,
            out_1x1pool=128,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, input):
        input = self.conv1(input)
        input = self.maxpool1(input)
        input = self.conv2(input)
        input = self.maxpool2(input)

        input = self.inception3a(input)
        input = self.inception3b(input)
        input = self.maxpool3(input)

        input = self.inception4a(input)
        input = self.inception4b(input)
        input = self.inception4c(input)
        input = self.inception4d(input)
        input = self.inception4e(input)
        input = self.maxpool4(input)

        input = self.inception5a(input)
        input = self.inception5b(input)

        input = self.avgpool(input)
        input = input.reshape(input.shape[0], -1)
        input = self.dropout(input)
        input = self.fc(input)
        return input


class InceptionBlock(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(
            in_channels=in_channels, out_channels=out_1x1, kernel_size=1
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=out_1x1pool, kernel_size=1),
        )

    def forward(self, input):
        # N x filterss x 64 x 64
        return torch.cat(
            [
                self.branch1(input),
                self.branch2(input),
                self.branch3(input),
                self.branch4(input),
            ],
            1,
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        return self.relu(self.batch_norm(self.conv(input)))
