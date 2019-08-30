# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from math import ceil
from torch import nn
from collections import OrderedDict

from .base import BackBoneCNN, NetworkInfo


class AlexNetConv(BackBoneCNN):
    """
    The standard AlexNet convolutional backbone.
    For more details, please refer to AlexNet paper:
    "ImageNet Classification with Deep Convolutional Neural Networks", NIPS 2012
    """

    num_blocks = 5
    blocks = dict(
        conv1=NetworkInfo(stride=4, channel=96, rf=15, size_func=lambda x: int(ceil(x / 4.0))),
        conv2=NetworkInfo(stride=8, channel=256, rf=39, size_func=lambda x: int(ceil(x / 8.0))),
        conv3=NetworkInfo(stride=8, channel=384, rf=55, size_func=lambda x: int(ceil(x / 8.0))),
        conv4=NetworkInfo(stride=8, channel=384, rf=71, size_func=lambda x: int(ceil(x / 8.0))),
        conv5=NetworkInfo(stride=8, channel=256, rf=87, size_func=lambda x: int(ceil(x / 8.0))),
    )

    def __init__(self, padding=True):
        super(AlexNetConv, self).__init__()
        if padding:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(3, 96, 11, stride=2, padding=5, bias=True)),
                ('relu', nn.ReLU()),
                ('pool', nn.MaxPool2d(3, stride=2, padding=1, dilation=1)),
                ('norm', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0))]))
            self.conv2 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(96, 256, 5, stride=1, padding=2, groups=2, bias=True)),
                ('relu', nn.ReLU()),
                ('pool', nn.MaxPool2d(3, stride=2, padding=1, dilation=1)),
                ('norm', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0))]))
            self.conv3 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=True)),
                ('relu', nn.ReLU())]))
            self.conv4 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=True)),
                ('relu', nn.ReLU())]))
            self.conv5 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=True)),
                ('relu', nn.ReLU())]))
        else:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(3, 96, 11, stride=2, padding=0, bias=True)),
                ('relu', nn.ReLU()),
                ('pool', nn.MaxPool2d(3, stride=2, padding=0, dilation=1)),
                ('norm', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0))]))
            self.conv2 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(96, 256, 5, stride=1, padding=0, groups=2, bias=True)),
                ('relu', nn.ReLU()),
                ('pool', nn.MaxPool2d(3, stride=2, padding=0, dilation=1)),
                ('norm', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0))]))
            self.conv3 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(256, 384, kernel_size=3, padding=0, bias=True)),
                ('relu', nn.ReLU())]))
            self.conv4 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(384, 384, kernel_size=3, padding=0, groups=2, bias=True)),
                ('relu', nn.ReLU())]))
            self.conv5 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(384, 256, kernel_size=3, padding=0, groups=2, bias=True)),
                ('relu', nn.ReLU())]))

