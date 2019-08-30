# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .alexnet import AlexNetConv

model_factory = dict(
    alexnet=AlexNetConv
)
