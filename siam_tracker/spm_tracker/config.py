# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from ..utils.collections import AttrDict

# ================= SPM-Tracker Configuration =======================

spm_tracker_cfgs = AttrDict()

spm_tracker_cfgs.NAME = 'spm_tracker'

# --------------- Coarse-matching configuration --------------------

spm_tracker_cfgs.MODEL = AttrDict()

spm_tracker_cfgs.MODEL.BACKBONE = 'alexnet'

# anchor setting for SiamRPN
spm_tracker_cfgs.MODEL.ANCHOR_SCALES = [8.0]
spm_tracker_cfgs.MODEL.ANCHOR_RATIOS = [0.33, 0.5, 1.0, 2.0, 3.0]

# context ratio in coarse matching stage
spm_tracker_cfgs.MODEL.Z_CONTEXT_AMOUNT = 0.5

# which layer is used for coarse-matching stage.
spm_tracker_cfgs.MODEL.FEAT_NAME = 'conv5'

# following SiamFC implementation, the xcorr output is scaled by a constant factor
# so that the training is more stable.
# ref: https://github.com/bilylee/SiamFC-TensorFlow
spm_tracker_cfgs.MODEL.CM_CLS_SCALE_FACTOR = 1e-3
spm_tracker_cfgs.MODEL.CM_BBOX_SCALE_FACTOR = 1e-3

# path of weight file
spm_tracker_cfgs.MODEL.CM_WEIGHTS_FILE = ''

spm_tracker_cfgs.MODEL.CM_HEAD_CHANNELS = 256

# the regional features are pooled into the fix-size by RoI Align op
spm_tracker_cfgs.MODEL.FM_ROI_POOL_SIZE = [6, 6]

# which layers are used for generating regional features
spm_tracker_cfgs.MODEL.FM_FEAT_NAME_LIST = ['conv4', 'conv2']

# whether applying for batch normalization before RoI Align.
# since the ordinary AlexNet does not have BN design,
# adding BN is expected to make training more stable.
spm_tracker_cfgs.MODEL.FM_ROI_FEAT_BN = True

# reduce the channels of concated reginal features (384+256=640) by convolutional op
# the POOL_CONVS hyper-param specifies how many conv layers are used (len(POOL_CONVS))
# and how many channels they have.
spm_tracker_cfgs.MODEL.FM_POOL_CONVS = [256]

# according to feature pyramid network in object detection. the detection head usually
# consists of several linear layers. we follow this design.
spm_tracker_cfgs.MODEL.FM_HIDDEN_LAYER_CHANNEL = 256
spm_tracker_cfgs.MODEL.FM_NUM_LINEAR_LAYERS = 2

# classificaiton types. by default it is target vs. non-target
spm_tracker_cfgs.MODEL.FM_NUM_CLASSES = 2

# predefined reciprocal of box variance. since the candidate boxes are quite close to the
# ground-truth, if we do not add any weights, the box regression loss is too small.
spm_tracker_cfgs.MODEL.FM_BBOX_VAR_INV = [5.0, 5.0, 10.0, 10.0]

# path of weight file
spm_tracker_cfgs.MODEL.FM_WEIGHTS_FILE = ''

# ------------------ Inference configuration ------------------------
spm_tracker_cfgs.INFERENCE = AttrDict()

# weight of score refinement in the second stage
spm_tracker_cfgs.INFERENCE.FM_SCORE_INFLUENCE = 0.5
# enable box refinement in the second stage?
spm_tracker_cfgs.INFERENCE.ENABLE_FM_BBOX_REFINE = True
# weight of box refinement in the second stage. (only works when ENABLE_FM_BBOX_REFINE == True)
spm_tracker_cfgs.INFERENCE.FM_BBOX_INFLUENCE = 3.0

# weight of cosine-windows
spm_tracker_cfgs.INFERENCE.WINDOWS_INFLUENCE = 0.4

# linear combination weight, the final tracking box is averaged by last frame result and current frame result
spm_tracker_cfgs.INFERENCE.LR = 0.5

# if the predication score is too low, tracker will directly output the last frame result
spm_tracker_cfgs.INFERENCE.MAX_SCORE_THRESHOLD = 0.1

# how many condidate boxes will be kept for the fine matching stage
spm_tracker_cfgs.INFERENCE.NUM_CANDIDATES = 9

# how many near-by boxes (to the last frame result) will be kept for the fine matching stage
spm_tracker_cfgs.INFERENCE.NUM_NEAR_CANDIDATES = 1

# the candidate boxes whose score is lower than max_score * SCORE_THRESH_FACTOR will be directly discarded
spm_tracker_cfgs.INFERENCE.SCORE_THRESH_FACTOR = 0.1

# NMS threshold in coarse matching stage.
spm_tracker_cfgs.INFERENCE.NMS_THRESHOLD = 0.5

# ------------------ Train configuration ------------------------

spm_tracker_cfgs.TRAIN = AttrDict()

# freeze some layers or not
spm_tracker_cfgs.TRAIN.FREEZE_RPN = False

# Coarse matching setting (SiamRPN here)
spm_tracker_cfgs.TRAIN.RPN_POSITIVE_OVERLAP = 0.6
spm_tracker_cfgs.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
spm_tracker_cfgs.TRAIN.RPN_FG_FRACTION = 0.5

# proposal settings
spm_tracker_cfgs.TRAIN.RPN_POST_NMS_TOP_N = 8
spm_tracker_cfgs.TRAIN.RPN_NMS_THRESH = 0.75

spm_tracker_cfgs.TRAIN.RPN_BATCHSIZE = 48

# Fine matching setting
spm_tracker_cfgs.TRAIN.FM_FG_THRESH = 0.5
spm_tracker_cfgs.TRAIN.FM_BG_THRESH = 0.5

# pre-trained model setting
spm_tracker_cfgs.TRAIN.INIT_CM_WEIGHTS_FILE = ""
spm_tracker_cfgs.TRAIN.INIT_FM_WEIGHTS_FILE = ""

# loss weight settings
spm_tracker_cfgs.TRAIN.CM_CLS_LOSS_WEIGHT = 1.0
spm_tracker_cfgs.TRAIN.CM_BBOX_LOSS_WEIGHT = 2.0
spm_tracker_cfgs.TRAIN.FM_CLS_LOSS_WEIGHT = 1.0
spm_tracker_cfgs.TRAIN.FM_BBOX_LOSS_WEIGHT = 1.0

# ---------------------------- End ----------------------------------
