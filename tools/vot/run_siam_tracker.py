# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import cv2
import torch
from torchvision.transforms.functional import to_tensor
import numpy as np
import os
from PIL import Image
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(root_dir, 'siam_tracker'))

from core.inference import SiamTracker
import benchmarks.vot.vot as vot
from benchmarks.vot.vot import Rectangle

from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list
import utils.boxes as ubox


def track(cfg_path, opts):

    # Setup tracker configuration
    merge_cfg_from_file(cfg_path)
    
    opts = opts.split(' ')
    if len(opts) > 0:
        merge_cfg_from_list(opts)

    device = torch.device('cuda:{}'.format(0))

    tracker = SiamTracker(device)
    
    def load_image(img_path, use_pil):
        if use_pil:
            pil_img = Image.open(img_path)
            if pil_img.mode == 'L':
                pil_img = pil_img.convert('RGB') # convert to RGB 3 channels if necessary
            im_tensor = to_tensor(pil_img)
        else:
            im = cv2.imread(img_path, cv2.IMREAD_COLOR)  # HxWxC
            im_tensor = torch.from_numpy(np.transpose(im, (2, 0, 1))).float()
        im_tensor = im_tensor.unsqueeze(0).to(device)  # 1*C*H*W
        return im_tensor

    # start to track
    handle = vot.VOT("polygon")
    Polygon = handle.region()

    box_cxcywh = vot.get_axis_aligned_bbox(Polygon)
    # convert to xyxy
    box_xyxy = ubox.xcycwh_to_xyxy(box_cxcywh)

    image_file = handle.frame()

    if not image_file:
        sys.exit(0)
    
    im_tensor = load_image(image_file, tracker.use_pil)
    tracker.tracker.init_tracker(im_tensor, box_xyxy)

    while True:
        image_file = handle.frame()
        if not image_file:
            break
        im_tensor = load_image(image_file, tracker.use_pil)
        box_xyxy = tracker.tracker.predict_next_frame(im_tensor, box_xyxy)
        box_xywh = ubox.xyxy_to_xywh(box_xyxy)

        handle.report(Rectangle(box_xywh[0], box_xywh[1], box_xywh[2], box_xywh[3]))