# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch

# xyxy denotes [x1, y1, x2, y2]
# xcycwh denotes [xc, yc, w, h]
# xywh denotes [x1, y1, w, h]

__all__ = ['xyxy_to_xywh', 'xywh_to_xyxy', 'xcycwh_to_xyxy', 'xyxy_to_xcycwh', 'xywh_to_xcycwh', 'xcycwh_to_xywh']


def _concat(x, y):
    """ Concat by the last dimension """
    if isinstance(x, np.ndarray):
        return np.concatenate((x, y), axis=-1)
    elif isinstance(x, torch.Tensor):
        return torch.cat([x, y], dim=-1)
    else:
        raise TypeError("unknown type '{}'".format(type(x)))


def xcycwh_to_xywh(xcycwh):
    """Convert [x_c y_c w h] box format to [x1, y1, w, h] format."""
    if isinstance(xcycwh, (list, tuple)):
        # Single box given as a list of coordinates
        assert not isinstance(xcycwh[0], (list, tuple))
        xc, yc = xcycwh[0], xcycwh[1]
        w = xcycwh[2]
        h = xcycwh[3]
        x1 = xc - w / 2.
        y1 = yc - h / 2.
        return [x1, y1, w, h]
    elif isinstance(xcycwh, (np.ndarray, torch.Tensor)):
        wh = xcycwh[..., 2:4]
        x1y1 = xcycwh[..., 0:2] - wh / 2.
        return _concat(x1y1, wh)
    else:
        raise TypeError('Argument xcycwh must be a list, tuple, or numpy array.')


def xywh_to_xcycwh(xywh):
    """Convert [x1 y1 w h] box format to [x_c y_c w h] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert not isinstance(xywh[0], (list, tuple))
        x1, y1 = xywh[0], xywh[1]
        xc = x1 + np.maximum(0., xywh[2] / 2.)
        yc = y1 + np.maximum(0., xywh[3] )
        return [xc, yc, xywh[2], xywh[3]]

    elif isinstance(xywh, (np.ndarray, torch.Tensor)):
        wh = xywh[..., 2:4]
        xcyc = xywh[..., 0:2] + wh / 2
        return _concat(xcyc, wh)
    else:
        raise TypeError('Argument xyxy must be a list, tuple, numpy array, or tensor.')


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert not isinstance(xywh[0], (list, tuple))
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2])
        y2 = y1 + np.maximum(0., xywh[3])
        return [x1, y1, x2, y2]
    elif isinstance(xywh, (np.ndarray, torch.Tensor)):
        x1y1 = xywh[..., 0:2]
        wh = xywh[..., 2:4]
        x2y2 = x1y1 + wh
        return _concat(x1y1, x2y2)
    else:
        raise TypeError('Argument xyxy must be a list, tuple, numpy array, or tensor.')


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert not isinstance(xyxy[0], (list, tuple))
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1
        h = xyxy[3] - y1
        return [x1, y1, w, h]
    elif isinstance(xyxy, (np.ndarray, torch.Tensor)):
        x1y1 = xyxy[..., 0:2]
        x2y2 = xyxy[..., 2:4]
        wh = x2y2 - x1y1
        return _concat(x1y1, wh)
    else:
        raise TypeError('Argument xyxy must be a list, tuple, numpy array, or tensor.')


def xcycwh_to_xyxy(xcycwh):
    """Convert [x_c y_c w h] box format to [x1, y1, x2, y2] format."""
    if isinstance(xcycwh, (list, tuple)):
        # Single box given as a list of coordinates
        assert not isinstance(xcycwh[0], (list, tuple))
        xc, yc = xcycwh[0], xcycwh[1]
        w = xcycwh[2]
        h = xcycwh[3]
        x1 = xc - w / 2.
        y1 = yc - h / 2.
        x2 = xc + w / 2.
        y2 = yc + h / 2.
        return [x1, y1, x2, y2]
    elif isinstance(xcycwh, (np.ndarray, torch.Tensor)):
        wh = xcycwh[..., 2:4]
        x1y1 = xcycwh[..., 0:2] - wh / 2.
        x2y2 = xcycwh[..., 0:2] + wh / 2.
        return _concat(x1y1, x2y2)
    else:
        raise TypeError('Argument xcycwh must be a list, tuple, or numpy array.')


def xyxy_to_xcycwh(xyxy):
    """Convert [x1 y1 x2, y2] box format to [x_c y_c w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert not isinstance(xyxy[0], (list, tuple))
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1
        h = xyxy[3] - y1
        x = (xyxy[0] + xyxy[2]) / 2.
        y = (xyxy[1] + xyxy[3]) / 2.
        return [x, y, w, h]
    elif isinstance(xyxy, (np.ndarray, torch.Tensor)):
        wh = xyxy[..., 2:4] - xyxy[..., 0:2]
        xy = (xyxy[..., 0:2] + xyxy[..., 2:4]) / 2.
        return _concat(xy, wh)
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')
