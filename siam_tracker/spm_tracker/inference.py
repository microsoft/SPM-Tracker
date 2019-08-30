# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch

from .fm_stage_wrapper import FM_Stage_Wrapper
from .modules.spm_tracker_cm import CM_Stage

from ..utils import (generate_anchors_on_response_map, rpn_logit_to_prob,
                     load_weights, crop_with_boxes)
from ..utils import boxes as ubox


class SPMTrackerInference(object):

    def __init__(self, model_device, img_norm, cfg):

        self.model_device = model_device
        self.cfg = cfg

        # create and initialize CM stage & FM stage
        self.cm = CM_Stage(cfg)
        load_weights(self.cm, self.cfg.MODEL.CM_WEIGHTS_FILE, verbose=False)
        self.cm.to(self.model_device)
        self.cm.eval()
        self.fm_wrapper = FM_Stage_Wrapper(cfg, model_device)

        self.z_size = self.cfg.INFERENCE.Z_SIZE
        self.x_size = self.cfg.INFERENCE.X_SIZE

        self.anchors = generate_anchors_on_response_map(
            z_size=self.z_size,
            z_response_size=self.cm.z_res_size,
            x_response_size=self.cm.x_res_size,
            model_stride=self.cm.stride,
            ratios=self.cfg.MODEL.ANCHOR_RATIOS,
            scales=self.cfg.MODEL.ANCHOR_SCALES)

        self.norm = img_norm

    def init_tracker(self, first_frame_img, current_box):
        """ Init the tracker (called in the first frame) """
        with torch.no_grad():
            search_img_tensor, z_box, x_box, scale_x = self.generate_search_image(first_frame_img, current_box)
            z_out = self.cm(z_image=search_img_tensor, x_image=None)

            # init fine matching stage
            z_box_in_search_img = \
                ubox.image_to_search_region(np.array([0.] + current_box, dtype=np.float32), x_box, scale_x)
            z_rois = torch.tensor(z_box_in_search_img).type_as(search_img_tensor).view(1, 5)
            z_feats = self.fm_wrapper.fm.feat_extractor(z_out, z_rois)
            self.fm_wrapper.set_template(z_feats)

    def predict_next_frame(self, img_tensor, current_box, vis=False):
        """ Predict the tracking result in given image
        Args:
            img_tensor: torch.Tensor in shape of [1, 3, H, W]
            current_box: list, in order of [x1, y1, x2, y2]
            vis: boolean, visualize tracking results or not (for debug)
        Returns:
            results: list in order of [x1, y1, x2, y2]
        """
        with torch.no_grad():
            search_img_tensor, z_box, x_box, scale_x = self.generate_search_image(img_tensor, current_box)
            det_boxes, x_out = self.generate_rpn_boxes(search_img_tensor, x_box, scale_x)

        proposed_boxes = self.generate_proposal_boxes(det_boxes, current_box)

        fm_scores, fm_boxes = self.fm_wrapper.get_similarity(x_out, proposed_boxes[:, 5:9])
        win_scores = self.calculate_window_scores(proposed_boxes)

        if self.cfg.INFERENCE.ENABLE_FM_BBOX_REFINE:
            fm_boxes_img = ubox.search_region_to_image(fm_boxes, x_box, scale_x)

            # calculate the weights
            v_influence = self.cfg.INFERENCE.FM_BBOX_INFLUENCE
            sum_scores = np.clip(proposed_boxes[:, 4] + fm_scores * v_influence, a_min=0.00001, a_max=99999.)
            _w = (proposed_boxes[:, 4] / sum_scores)[:, np.newaxis]

            proposed_boxes[:, :4] = proposed_boxes[:, :4] * _w + fm_boxes_img[:, :4] * (1. - _w)
            proposed_boxes[:, 5:9] = proposed_boxes[:, 5:9] * _w + fm_boxes[:, :4] * (1. - _w)

        cm_scores = proposed_boxes[:, 4]
        pred_scores = cm_scores * (1 - self.cfg.INFERENCE.FM_SCORE_INFLUENCE) + \
                      fm_scores * self.cfg.INFERENCE.FM_SCORE_INFLUENCE

        final_scores = pred_scores * (1 - self.cfg.INFERENCE.WINDOWS_INFLUENCE) + \
                       win_scores * self.cfg.INFERENCE.WINDOWS_INFLUENCE

        max_score_idx = np.argmax(final_scores)

        best_box = proposed_boxes[max_score_idx, :4].tolist()
        best_score = pred_scores[max_score_idx]

        if best_score < self.cfg.INFERENCE.MAX_SCORE_THRESHOLD:
            best_box = current_box
        else:
            lr = best_score * self.cfg.INFERENCE.LR
            best_box = self.linear_inter(best_box, current_box, lr, img_tensor.size(2), img_tensor.size(3))

        return best_box

    def generate_rpn_boxes(self, img_tensor, x_crop_box, scale_x):
        """ Forward SiamRPN network and get the predicted boxes.
        Args:
            img_tensor: torch.Tensor in shape of [1, 3, x_size, x_size], search image tensor
        Returns:
            det_boxes: numpy.ndarray in shape of [x_res_size * x_res_size * num_anchors, 9]. The first four values
                       indicate the coordinates in source image domain. The following one value denotes the rpn score.
                       the last four values are coordinates in search region domain.
            x_out: dict. Convolutional features of search image.

        """
        z_out, x_out, rpn_cls_logit, rpn_bbox_delta = self.cm(z_image=None, x_image=img_tensor)
        rpn_cls_prob = \
            rpn_logit_to_prob(rpn_cls_logit)
        rpn_cls_prob = rpn_cls_prob[:, rpn_cls_prob.size(1) // 2:, :, :]

        rpn_bbox_delta = rpn_bbox_delta.data.cpu().numpy()[0].transpose((1, 2, 0)).reshape(-1, 4)
        rpn_score = rpn_cls_prob.data.cpu().numpy()[0].transpose((1, 2, 0)).flatten()
        rpn_bbox_pred = ubox.bbox_transform(self.anchors, rpn_bbox_delta)

        # transform to original image domain
        rpn_bbox_pred_img = ubox.search_region_to_image(rpn_bbox_pred, x_crop_box, scale_x)

        # combine the coordinates in two domains
        det_boxes = np.hstack((rpn_bbox_pred_img, rpn_score[:, np.newaxis], rpn_bbox_pred))

        return det_boxes, x_out

    def generate_proposal_boxes(self, rpn_boxes, current_box):
        """ Propose some candidate boxes.
        Args:
            rpn_boxes: numpy.ndarray in shape of [N, 9], see 'generate_rpn_boxes' function for more details.
            current_box: list in order of [x1, y1, x2, y2]
        Returns:
            proposed_boxes: numpy.ndarray in shape of [K, 9], the proposal results
        """

        num_near_candidates = self.cfg.INFERENCE.NUM_NEAR_CANDIDATES
        num_candidates = self.cfg.INFERENCE.NUM_CANDIDATES

        current_box_xyxy = np.array(current_box, dtype=np.float32)

        # Step 1. Find the candidates which are closest to the current box
        if num_near_candidates > 0:
            overlaps = ubox.bbox_overlaps(np.ascontiguousarray(rpn_boxes[:, :4], dtype=np.float32),
                                          np.array(current_box_xyxy, dtype=np.float32).reshape(1, 4)).flatten()
            is_near_box = (overlaps > 0.80)
            near_inds = np.where(is_near_box)[0]

            # if there exist a few boxes which is close to current box
            # we will select the one with the highest rpn score.
            if len(near_inds) > num_near_candidates:
                ov_sorted_inds = near_inds[np.argsort(rpn_boxes[near_inds, 4])[::-1]]
                near_boxes = rpn_boxes[ov_sorted_inds[:num_near_candidates], :]

                other_inds = np.where(~is_near_box)[0]
                other_inds = np.concatenate((ov_sorted_inds[num_near_candidates:], other_inds))
                proposed_boxes = rpn_boxes[other_inds, :]
            else:
                ov_sorted_inds = np.argsort(overlaps)[::-1]
                near_boxes = rpn_boxes[ov_sorted_inds[:num_near_candidates], :]
                proposed_boxes = rpn_boxes[ov_sorted_inds[num_near_candidates:], :]
        else:
            proposed_boxes = rpn_boxes

        # Step 2. filter some box whose score is too low.
        max_score = np.max(proposed_boxes[:, 4])
        kept_inds = np.where(proposed_boxes[:, 4] >= self.cfg.INFERENCE.SCORE_THRESH_FACTOR * max_score)[0]
        proposed_boxes = proposed_boxes[kept_inds, :]

        # Step 3. apply for NMS
        if proposed_boxes.shape[0] > 1:
            kept_inds = ubox.nms(np.ascontiguousarray(proposed_boxes[:, :5], dtype=np.float32),
                                 self.cfg.INFERENCE.NMS_THRESHOLD)
            proposed_boxes = proposed_boxes[kept_inds, :]
            sort_inds = np.argsort(proposed_boxes[:, 4])[::-1]
            proposed_boxes = proposed_boxes[sort_inds, :]

            _k = min(proposed_boxes.shape[0], num_candidates - num_near_candidates)
            proposed_boxes = proposed_boxes[:_k, :]

        if num_near_candidates > 0:
            proposed_boxes = np.vstack((proposed_boxes, near_boxes))

        return proposed_boxes

    def calculate_window_scores(self, det_boxes):
        """ Add cosine windows penalty.
        Args:
            det_boxes: numpy.ndarray
        """
        det_boxes_on_x_xcycwh = ubox.xyxy_to_xcycwh(det_boxes[:, 5:9])
        center_dist = det_boxes_on_x_xcycwh[:, 0:2] / (self.x_size - 1)
        center_dist = np.maximum(np.minimum(center_dist, 1.0), 0.0)
        win_scores_2d = 0.5 - 0.5 * np.cos(2. * np.pi * center_dist)
        win_scores = win_scores_2d[:, 0] * win_scores_2d[:, 1]
        return win_scores

    def linear_inter(self, det_box, current_box, lr, im_height, im_width):
        """ To make tracking more stable, the final result is a linear combination between
            predicted result and last frame result.

        Args:
            det_box: list in order of [x1, y1, x2, y2], predicted result
            current_box: list in order of [x1, y1, x2, y2], predicted result
            lr: linear combination weight
        Results:
            refine_box: list in order of [x1, y1, x2, y2]

        """
        _, _, current_w, current_h = ubox.xyxy_to_xcycwh(current_box)
        refine_box = ubox.xyxy_to_xcycwh(det_box)
        refine_box[2] = refine_box[2] * lr + current_w * (1 - lr)
        refine_box[3] = refine_box[3] * lr + current_h * (1 - lr)

        refine_box[2] = max(10, refine_box[2])
        refine_box[3] = max(10, refine_box[3])
        refine_box[0] = max(0, min(refine_box[0], im_width - (refine_box[2]) / 2.))
        refine_box[1] = max(0, min(refine_box[1], im_height - (refine_box[3]) / 2.))

        refine_box = ubox.xcycwh_to_xyxy(refine_box)

        return refine_box

    def generate_search_image(self, img_tensor, current_box):
        # generate search regions
        z_box, x_box, scale_x = ubox.generate_search_region_boxes(
            boxes=current_box,
            z_size=self.z_size,
            x_size=self.x_size,
            context_amount=self.cfg.MODEL.Z_CONTEXT_AMOUNT)
        search_img_tensor = \
            crop_with_boxes(img_tensor, x_box, self.x_size, self.x_size)

        search_img_tensor = self.norm(search_img_tensor)

        return search_img_tensor, z_box, x_box, scale_x
