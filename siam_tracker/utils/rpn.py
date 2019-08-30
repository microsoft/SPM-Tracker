# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch


def rpn_reshape(x, d):
    input_shape = x.size()
    if d > 1:
        return x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3])
    else:
        return x.view(
            input_shape[0],
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3])


def rpn_logit_to_prob(rpn_cls_logit, use_softmax=True):
    if use_softmax:
        rpn_cls_logit_reshape = rpn_reshape(rpn_cls_logit, 2)
        rpn_cls_prob_reshape = torch.nn.functional.softmax(rpn_cls_logit_reshape, dim=1)
        rpn_cls_prob = rpn_reshape(rpn_cls_prob_reshape, rpn_cls_logit.size(1))
    else:
        rpn_cls_prob = torch.sigmoid(rpn_cls_logit)

    return rpn_cls_prob


def generate_anchors_on_response_map(z_size, z_response_size, x_response_size, model_stride, ratios, scales):
    basic_anchors = generate_anchors(model_stride, ratios=ratios, scales=scales)

    output_response_size = x_response_size - z_response_size + 1

    shift_x = np.arange(0, output_response_size) * model_stride
    shift_y = np.arange(0, output_response_size) * model_stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    A = basic_anchors.shape[0]
    K = shifts.shape[0]
    anchors = basic_anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    anchors += (z_size / 2)

    return anchors


def generate_anchors(base_size=8, ratios=[0.5, 1, 2],
                     scales=[8.]):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    scales = np.array(scales)
    ratios = np.array(ratios)
    base_anchor = np.array([1, 1, base_size, base_size]) - 1 - (base_size - 1) / 2.
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
