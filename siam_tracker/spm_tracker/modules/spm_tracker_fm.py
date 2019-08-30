# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Fine matching stage """

import torch
import torch.nn.functional as F

from ...models import model_factory
from ...ops.crop_and_resize import CropAndResizeFunction


class FeatExtractor(torch.nn.Module):

    def __init__(self, cfg):
        """ extractor regional features from the shared conv layers """
        super(FeatExtractor, self).__init__()
        self.cfg = cfg

        self.out_height, self.out_width = self.cfg.MODEL.FM_ROI_POOL_SIZE

        model_info = model_factory[self.cfg.MODEL.BACKBONE]
        self.feat_name_list = self.cfg.MODEL.FM_FEAT_NAME_LIST
        self.nc_list = model_info.infer_channels(self.feat_name_list)
        self.stride_list = model_info.infer_strides(self.feat_name_list)
        self.out_channel = int(sum(self.nc_list))

        self.use_bn = self.cfg.MODEL.FM_ROI_FEAT_BN
        if self.use_bn:
            self.bn_list = torch.nn.ModuleList(
                [torch.nn.BatchNorm2d(nc) for nc in self.nc_list]
            )

    def forward(self, feats, rois, roi_inds=None):

        if roi_inds is None:
            roi_boxes = rois[:, 1:5].contiguous()
            roi_inds = rois[:, 0].contiguous()
        else:
            roi_boxes = rois.contiguous()
            roi_inds = roi_inds.contiguous()

        roi_feats = []
        for ix, feat_name in enumerate(self.feat_name_list):
            if self.use_bn:
                i_feat = self.bn_list[ix](feats[feat_name])
            else:
                i_feat = feats[feat_name]

            i_roi_boxes = roi_boxes / self.stride_list[ix]
            i_roi_feat = CropAndResizeFunction(self.out_height, self.out_width, has_normed=False)(
                i_feat, i_roi_boxes, roi_inds)
            roi_feats.append(i_roi_feat)

        roi_feats = torch.cat(roi_feats, dim=1)
        return roi_feats


class SiamFeatFusion(torch.nn.Module):

    def __init__(self, cfg, in_channel):
        super(SiamFeatFusion, self).__init__()

        self.cfg = cfg

        self.in_height, self.in_width = self.cfg.MODEL.FM_ROI_POOL_SIZE
        self.channel = in_channel * 2  # concat Siamese features, thus we have double channels

        self.num_conv = len(self.cfg.MODEL.FM_POOL_CONVS)  # how many 1x1 convolution layers we used
        if self.num_conv > 0:
            self.roi_conv = torch.nn.Sequential()
            for i in range(self.num_conv):
                c = self.cfg.MODEL.FM_POOL_CONVS[i]
                self.roi_conv.add_module('roi_conv{}'.format(i),
                                         torch.nn.Conv2d(self.channel, c, 3, stride=1, padding=1))
                self.channel = c
                self.roi_conv.add_module('roi_relu{}'.format(i),
                                         torch.nn.ReLU(inplace=True))

        self.hidden_channel = self.cfg.MODEL.FM_HIDDEN_LAYER_CHANNEL

        self.num_linear = self.cfg.MODEL.FM_NUM_LINEAR_LAYERS
        assert self.num_linear > 0, 'Number of linear layer should greater than 0.'

        self.linear_layers = torch.nn.Sequential()
        for i in range(self.num_linear):
            if i == 0:
                in_c = self.channel * self.in_height * self.in_width
            else:
                in_c = self.hidden_channel

            self.linear_layers.add_module(
                'linear{}'.format(i), torch.nn.Linear(in_c, self.hidden_channel))
            self.linear_layers.add_module(
                'relu{}'.format(i), torch.nn.ReLU())

    def forward(self, z_feat, x_feat):
        num_x = x_feat.size(0)
        num_z, zc, zh, zw = z_feat.size()
        if num_z == 1 and num_x > 1:
            fusion_feat = torch.cat((z_feat.expand(num_x, -1, -1, -1), x_feat), dim=1).contiguous()
        else:
            fusion_feat = torch.cat((z_feat, x_feat), dim=1).contiguous()

        if self.num_conv > 0:
            fusion_feat = self.roi_conv(fusion_feat)
        fusion_feat = fusion_feat.view(num_x, -1)
        fusion_feat = self.linear_layers(fusion_feat)

        return fusion_feat


class FM_Stage_Head(torch.nn.Module):

    def __init__(self, cfg):
        super(FM_Stage_Head, self).__init__()

        self.cfg = cfg

        self.feat_extractor = FeatExtractor(self.cfg)
        self.siam_fusion = SiamFeatFusion(self.cfg, self.feat_extractor.out_channel)

        self.num_classes = self.cfg.MODEL.FM_NUM_CLASSES

        self.rcnn_cls = torch.nn.Linear(self.siam_fusion.hidden_channel, self.num_classes)
        self.rcnn_bbox = torch.nn.Linear(self.siam_fusion.hidden_channel, 4)
