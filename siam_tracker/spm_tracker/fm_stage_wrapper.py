# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn.functional as F

from .modules.spm_tracker_fm import FM_Stage_Head

from ..utils import load_weights, bbox_transform


class FM_Stage_Wrapper(object):

    def __init__(self, model_cfg, model_device):
        self.model_cfg = model_cfg
        self.model_device = model_device

        self.fm = FM_Stage_Head(model_cfg)
        load_weights(self.fm, model_cfg.MODEL.FM_WEIGHTS_FILE, verbose=False)
        self.fm.to(self.model_device)
        self.fm.eval()

        self.feat_height, self.feat_width = model_cfg.MODEL.FM_ROI_POOL_SIZE
        self.feat_channel = self.fm.feat_extractor.out_channel

        # setup template embeddings
        self.temp_embedding = torch.zeros(1, self.feat_channel, self.feat_height, self.feat_width,
                                          device=model_device, dtype=torch.float32)

    def set_template(self, z_feats):
        self.temp_embedding[...] = z_feats.view(1, self.feat_channel, self.feat_height, self.feat_width)

    def get_similarity(self, x_out, proposals):
        """
        Args:
            x_feats: list of torch.Tensor
            proposal: numpy array
        """

        with torch.no_grad():
            num_proposals = proposals.shape[0]
            x_rois_np = np.hstack((np.zeros((num_proposals, 1)), proposals))
            x_rois = torch.from_numpy(x_rois_np).float().to(self.model_device)
            sim_cls_logit, sim_box_deltas = \
                self.extract_sim_mat(x_out=x_out, x_rois=x_rois)
            # sim_cls_logit [num_rois, 2]
            # sim_box_deltas [num_rois, 4]
            sim_prob = F.softmax(sim_cls_logit, dim=1)  # [num_rois, 2]

        sim_score = sim_prob.cpu().numpy()[:, 1]
        bbox_delta = sim_box_deltas.data.cpu().numpy()

        bbox_pred = bbox_transform(proposals[:, :4], bbox_delta, weights=self.model_cfg.MODEL.FM_BBOX_VAR_INV)
        return sim_score, bbox_pred

    def extract_sim_mat(self, x_out, x_rois):
        x_feats = self.fm.feat_extractor(x_out, x_rois)
        fusion_feat = self.fm.siam_fusion(self.temp_embedding, x_feats)
        sim_mat = self.fm.rcnn_cls(fusion_feat)  # N x 2
        bbox_mat = self.fm.rcnn_bbox(fusion_feat)

        return sim_mat, bbox_mat
