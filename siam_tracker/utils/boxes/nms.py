# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import numpy as np
from . import cython_nms as _nms_np_backend


def nms(dets, nms_thresh=0.7):
    if isinstance(dets, np.ndarray):
        keep_inds = _nms_np_backend.nms(dets, nms_thresh)
    elif isinstance(dets, torch.Tensor):
        raise NotImplementedError("Not support torch.Tensor")
    else:
        raise TypeError('Unknown type {}'.format(type(dets)))

    return keep_inds
