# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F

from ...models import model_factory
from ...utils import center_crop_tensor


class CM_Stage_Head(torch.nn.Module):

    def __init__(self, cfg):
        super(CM_Stage_Head, self).__init__()
        self.cfg = cfg
        model_info = model_factory[self.cfg.MODEL.BACKBONE]
        feat_name = self.cfg.MODEL.FEAT_NAME  # by default 'conv5'
        conv_channels = model_info.infer_channels(feat_name)

        self.num_channels = self.cfg.MODEL.CM_HEAD_CHANNELS
        self.num_anchors = len(self.cfg.MODEL.ANCHOR_SCALES) * len(self.cfg.MODEL.ANCHOR_RATIOS)

        self.z_cls_conv = nn.Conv2d(conv_channels, self.num_channels * self.num_anchors * 2, 3)
        self.z_bbox_conv = nn.Conv2d(conv_channels, self.num_channels * self.num_anchors * 4, 3)

        self.x_cls_conv = nn.Sequential(nn.Conv2d(conv_channels, self.num_channels, 3))
        self.x_bbox_conv = nn.Sequential(nn.Conv2d(conv_channels, self.num_channels, 3))

        self.bbox_refine = nn.Conv2d(self.num_anchors * 4, self.num_anchors * 4, 1)

        self.cls_scale_factor = self.cfg.MODEL.CM_CLS_SCALE_FACTOR
        self.bbox_scale_factor = self.cfg.MODEL.CM_BBOX_SCALE_FACTOR
        # Since the bbox branch has an refine step, we don't need to add another bias term in bbox branch
        self.cls_bias = nn.Parameter(torch.zeros(1, self.num_anchors * 2, 1, 1), requires_grad=False)

        z_size = self.cfg.INFERENCE.Z_SIZE
        z_res_size = model_info.infer_size(feat_name, z_size, padding=False)
        param_size = z_res_size - 2  # processed by a 3x3 conv, thus the size reduces 2 pixels.
        self.z_cls_weights = nn.Parameter(torch.zeros(self.num_anchors * 2, self.num_channels, param_size, param_size),
                                          requires_grad=False)
        self.z_bbox_weights = nn.Parameter(torch.zeros(self.num_anchors * 4, self.num_channels, param_size, param_size),
                                           requires_grad=False)

    def forward(self, z_embedding=None, x_embedding=None):
        if z_embedding is not None:
            assert z_embedding.shape[0] == 1, 'Support batch=1 only. (for inference).'
            z_cls_branch = self.z_cls_conv(z_embedding)
            z_bbox_branch = self.z_bbox_conv(z_embedding)

            self.z_cls_weights.data[:] = z_cls_branch.view(self.z_cls_weights.size()) * self.cls_scale_factor
            self.z_bbox_weights.data[:] = z_bbox_branch.view(self.z_bbox_weights.size()) * self.bbox_scale_factor
            return None, None

        x_cls_branch = self.x_cls_conv(x_embedding)
        x_bbox_branch = self.x_bbox_conv(x_embedding)

        rpn_cls_logit = F.conv2d(x_cls_branch, self.z_cls_weights) + self.cls_bias
        rpn_bbox_delta = F.conv2d(x_bbox_branch, self.z_bbox_weights)
        rpn_bbox_delta = self.bbox_refine(rpn_bbox_delta)

        return rpn_cls_logit, rpn_bbox_delta


class CM_Stage(torch.nn.Module):

    def __init__(self, cfg):

        super(CM_Stage, self).__init__()

        self.cfg = cfg

        model_info = model_factory[self.cfg.MODEL.BACKBONE]
        self.feat_name = self.cfg.MODEL.FEAT_NAME  # by default 'conv5'
        self.stride = model_info.infer_strides(self.feat_name)

        z_size = self.cfg.INFERENCE.Z_SIZE
        x_size = self.cfg.INFERENCE.X_SIZE
        self.z_res_size = model_info.infer_size(self.feat_name, z_size, padding=False)
        self.x_res_size = model_info.infer_size(self.feat_name, x_size, padding=False)

        self.z_net = model_info(padding=True)  # the backbone networks

        # construct rpn head for coarse matching stage
        self.rpn_head = CM_Stage_Head(cfg)

    def forward(self, z_image=None, x_image=None):

        if z_image is not None:
            z_out = self.z_net(z_image)
            z_embeddings = center_crop_tensor(z_out[self.feat_name], self.z_res_size, self.z_res_size)
        else:
            z_out = None
            z_embeddings = None

        if x_image is not None:
            x_out = self.z_net(x_image)
            x_embeddings = center_crop_tensor(x_out[self.feat_name], self.x_res_size, self.x_res_size)
        else:
            x_out = None
            x_embeddings = None

        rpn_cls_logit, rpn_bbox_delta = self.rpn_head(z_embedding=z_embeddings, x_embedding=x_embeddings)

        # if x_image is None, it means that only forwarding z_image is reset the template features in rpn_head.
        if x_image is None:
            return z_out

        return z_out, x_out, rpn_cls_logit, rpn_bbox_delta
