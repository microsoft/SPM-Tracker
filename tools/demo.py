# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import _init_paths
import os
import torch
import cv2
import numpy as np
import time
import argparse

from siam_tracker.core.inference import SiamTracker
from siam_tracker.core.config import merge_cfg_from_list, merge_cfg_from_file

global vis_img
global point1, point2


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Inference with Siamese Network based framework')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use, using gpu[0] by default',
                        type=int,
                        default=0)
    parser.add_argument('--video', dest='video',
                        help='video which is expected to load',
                        type=str,
                        default='data/demo/boy.avi', )
    parser.add_argument('--cfg', dest='config_file',
                        help='config file path',
                        type=str,
                        default='configs/spm_alexnet.yaml')
    parser.add_argument('opts',
                        help='all other options',
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = parser.parse_args()
    return args


def assign_init_box():
    global vis_img, point1, point2
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', vis_img)
    cv2.waitKey(0)

    x1, x2 = min(point1[0], point2[0]), max(point1[0], point2[0])
    y1, y2 = min(point1[1], point2[1]), max(point1[1], point2[1])
    init_box = [x1, y1, x2, y2]
    return init_box


def on_mouse(event, x, y, flags, param):
    """ Drawing rectangle, from Xiaoqiang Zhou """
    global vis_img, point1, point2
    img2 = vis_img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # left click
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 3)  # green
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # left drag
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 3)  # blue
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # left release
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 3)  # red
        cv2.imshow('image', img2)
        cv2.destroyAllWindows()


def img2tensor(img, device):
    """ Convert numpy.ndarry to torch.Tensor """
    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().to(device)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def draw_bbox(img, box):
    box_int = [int(round(_)) for _ in box]
    img = cv2.rectangle(img, (box_int[0], box_int[1]), (box_int[2], box_int[3]), color=(0, 255, 0), thickness=2)
    return img


if __name__ == '__main__':
    global vis_img
    global point1, point2

    args = parse_args()
    assert os.path.exists(args.video), "Cannot find video {}".format(args.video)

    # setup tracker configuration
    cfg_path = args.config_file
    if cfg_path != "":
        merge_cfg_from_file(cfg_path)
    if len(args.opts) > 0:
        merge_cfg_from_list(args.opts)

    # setup devices
    if args.gpu_id < 0:
        # use CPU if device id < 0
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    # setup tracker
    model = SiamTracker(device)
    # load video
    cap = cv2.VideoCapture(args.video)
    if cap.isOpened():
        _, frame = cap.read()
        vis_img = frame.copy()
    else:
        raise IOError("Cannot read frames from {}".format(args.videos))
    init_box = assign_init_box()
    img_tensor = img2tensor(frame, device)

    # init tracker
    with torch.no_grad():
        model.tracker.init_tracker(img_tensor, init_box)
        current_box = init_box
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break
            if current_box is not None:
                img_tensor = img2tensor(frame, device)
                current_box = model.tracker.predict_next_frame(img_tensor, current_box)
                if current_box is not None:
                    frame = draw_bbox(frame, current_box)
            cv2.imshow('image', frame)
            cv2.waitKey(40)

    cap.release()
