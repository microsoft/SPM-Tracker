# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch

from . import xcycwh_to_xyxy, xyxy_to_xcycwh


def image_to_search_region(boxes, crop_box, scale_x):
    ret = boxes.copy()
    if len(boxes.shape) == 1:
        st = 0 if boxes.shape[0] == 4 else 1
        ret[st + 0:st + 4:2] -= crop_box[0]
        ret[st + 1:st + 4:2] -= crop_box[1]
        ret[st + 0:st + 4] *= scale_x
    else:
        st = 0 if boxes.shape[1] == 4 else 1
        ret[:, st + 0:st + 4:2] -= crop_box[0]
        ret[:, st + 1:st + 4:2] -= crop_box[1]
        ret[:, st + 0:st + 4] *= scale_x

    return ret


def search_region_to_image(boxes, crop_box, scale_x):
    ret = boxes.copy()
    if len(boxes.shape) == 1:
        st = 0 if boxes.shape[0] == 4 else 1
        ret[st + 0:st + 4] /= scale_x
        ret[st + 0:st + 4:2] += crop_box[0]
        ret[st + 1:st + 4:2] += crop_box[1]
    else:
        st = 0 if boxes.shape[1] == 4 else 1
        ret[:, st + 0:st + 4] /= scale_x
        ret[:, st + 0:st + 4:2] += crop_box[0]
        ret[:, st + 1:st + 4:2] += crop_box[1]

    return ret


def generate_crop_boxes(boxes, context_amount=0.5, keep_ar=True):
    boxes_wh = xyxy_to_xcycwh(boxes)

    if keep_ar:
        ctx_size = (boxes_wh[:, 2] + boxes_wh[:, 3]) * context_amount
        wc_z = boxes_wh[:, 2] + ctx_size
        hc_z = boxes_wh[:, 3] + ctx_size
        if isinstance(boxes_wh, np.ndarray):
            s_z = np.sqrt(wc_z * hc_z)
        else:
            s_z = torch.sqrt(wc_z * hc_z)
        boxes_wh[:, 2] = s_z
        boxes_wh[:, 3] = s_z
    else:
        boxes_wh[:, 2] = boxes_wh[:, 2] + boxes_wh[:, 2] * context_amount * 2
        boxes_wh[:, 3] = boxes_wh[:, 3] + boxes_wh[:, 3] * context_amount * 2
    crop_boxes = xcycwh_to_xyxy(boxes_wh)

    return crop_boxes


def generate_search_region_boxes(boxes, z_size=127, x_size=255, context_amount=0.5):
    """ Generate the crop boxes for tracking

    Parameters:
        boxes: list/numpy.ndarry/torch.Tensor , in order of [x1, y1, x2, y2]

    Returns:
        box_z: template crop box, same type with boxes
        box_x: search region crop box, same type with boxes
        scale_x: scale factor to cropped domain
    """

    boxes_wh = xyxy_to_xcycwh(boxes)
    if isinstance(boxes, list):
        assert not isinstance(boxes[0], list), 'One dimentional list supported only'
        cx, cy, w, h = boxes_wh
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = z_size / s_z

        d_search = (x_size - z_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        scale_x = x_size / s_x

        box_z_wh = [cx, cy, s_z, s_z]
        box_x_wh = [cx, cy, s_x, s_x]

    elif isinstance(boxes, np.ndarray):
        ctx_size = (boxes_wh[:, 2] + boxes_wh[:, 3]) * context_amount
        wc_z = boxes_wh[:, 2] + ctx_size
        hc_z = boxes_wh[:, 3] + ctx_size
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = z_size / s_z

        d_search = (x_size - z_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        scale_x = x_size / s_x

        box_z_wh = np.hstack((boxes_wh[:, :2], s_z[:, np.newaxis], s_z[:, np.newaxis]))
        box_x_wh = np.hstack((boxes_wh[:, :2], s_x[:, np.newaxis], s_x[:, np.newaxis]))

    elif isinstance(boxes, torch.Tensor):
        ctx_size = (boxes_wh[:, 2] + boxes_wh[:, 3]) * context_amount
        wc_z = boxes_wh[:, 2] + ctx_size
        hc_z = boxes_wh[:, 3] + ctx_size
        s_z = torch.sqrt(wc_z * hc_z)
        scale_z = z_size / s_z

        d_search = (x_size - z_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        scale_x = x_size / s_x

        box_z_wh = torch.cat((boxes_wh[:, :2], s_z.view(-1, 1), s_z.view(-1, 1)), dim=1).contiguous()
        box_x_wh = torch.cat((boxes_wh[:, :2], s_x.view(-1, 1), s_x.view(-1, 1)), dim=1).contiguous()

    else:
        raise NotImplementedError

    box_z = xcycwh_to_xyxy(box_z_wh)
    box_x = xcycwh_to_xyxy(box_x_wh)

    return box_z, box_x, scale_x
