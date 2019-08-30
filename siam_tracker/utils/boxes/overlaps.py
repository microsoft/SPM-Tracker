# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import numpy as np
from .cython_bbox import bbox_overlaps as bbox_overlaps_np


def bbox_overlaps(anchors, gt_boxes):
    if isinstance(anchors, np.ndarray):
        return bbox_overlaps_np(anchors, gt_boxes)
    elif isinstance(anchors, torch.Tensor):
        raise NotImplementedError
    else:
        raise ValueError("Unknown type of 'anchors' {}".format(type(anchors)))
