# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch

__all__ = ['bbox_transform_inv', 'bbox_transform']

BBOX_XFORM_CLIP = float(np.log(255. / 32.))


def bbox_transform_inv(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.
    """
    is_torch = isinstance(boxes, torch.Tensor)

    ex_widths = boxes[:, 2] - boxes[:, 0]
    ex_heights = boxes[:, 3] - boxes[:, 1]
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    if is_torch:
        gt_widths = torch.clamp(gt_boxes[:, 2] - gt_boxes[:, 0], min=1e-3)
        gt_heights = torch.clamp(gt_boxes[:, 3] - gt_boxes[:, 1], min=1e-3)
    else:
        gt_widths = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 1e-3)
        gt_heights = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1], 1e-3)

    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights

    if is_torch:
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.cat((targets_dx.unsqueeze(1),
                             targets_dy.unsqueeze(1),
                             targets_dw.unsqueeze(1),
                             targets_dh.unsqueeze(1)), dim=1)

    else:
        targets_dw = ww * np.log(gt_widths / ex_widths)
        targets_dh = wh * np.log(gt_heights / ex_heights)

        targets = np.vstack((targets_dx, targets_dy, targets_dw,
                             targets_dh)).transpose()
    return targets


def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """

    is_torch = isinstance(boxes, torch.Tensor)

    if len(boxes) == 0:
        if is_torch:
            return torch.zeros_like(boxes)
        else:
            return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    if not is_torch:
        boxes = boxes.astype(deltas.dtype, copy=False)
    else:
        boxes = boxes.to(dtype=deltas.dtype, device=deltas.device)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    if not is_torch:
        dw_invalid = (np.isnan(dw) | np.isinf(dw))
        dh_invalid = (np.isnan(dh) | np.isinf(dh))
        dw[dw_invalid] = BBOX_XFORM_CLIP
        dh[dh_invalid] = BBOX_XFORM_CLIP
        dw = np.minimum(dw, BBOX_XFORM_CLIP)
        dh = np.minimum(dh, BBOX_XFORM_CLIP)

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry) (removed -1 2018.09.28)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry) (removed -1 2018.09.28)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    else:
        dw_invalid = torch.isnan(dw)
        dh_invalid = torch.isnan(dh)
        dw[dw_invalid] = 0
        dh[dh_invalid] = 0
        dw = torch.clamp(dw, min=-BBOX_XFORM_CLIP, max=BBOX_XFORM_CLIP)
        dh = torch.clamp(dh, min=-BBOX_XFORM_CLIP, max=BBOX_XFORM_CLIP)

        pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
        pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
        pred_w = torch.exp(dw) * widths.unsqueeze(1)
        pred_h = torch.exp(dh) * heights.unsqueeze(1)

        pred_boxes = torch.zeros_like(deltas)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry) (removed -1 2018.09.28)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry) (removed -1 2018.09.28)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
