# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import _init_paths

import os
import numpy as np
import argparse
import yaml
import logging
import sys
import time
import collections
import torch

import siam_tracker.benchmarks.lasot as lasot

from siam_tracker.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list
from siam_tracker.core.inference import SiamTracker

import siam_tracker.utils.boxes as ubox
from siam_tracker.utils.misc import mkdir_if_not_exists

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Script to run tracker')
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu device index. negative index will lead to cpu mode',
                        default=0,
                        type=int)
    parser.add_argument('--cfg', dest='config_file',
                        help='config file path',
                        default='configs/spm_tracker/alexnet_c42_otb.yaml',
                        type=str)
    parser.add_argument('--overwrite', dest='overwrite',
                        help='overwrite',
                        default=0,
                        type=int)
    parser.add_argument('opts',
                        help='all options',
                        default=None,
                        nargs=argparse.REMAINDER
    )

    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)
    
    args = parser.parse_args()
    return args


def run_SiamTracker(seq, device):

    tracker = SiamTracker(device)

    tic = time.clock()
    frames = seq['frames']
    init_bb = seq['init_rect']
    trajectory_py = tracker.track(init_box=init_bb, input_str=frames, box_format='tlbr')
    duration = time.clock() - tic

    # trajectory = [t[0], t[1], t[2] - t[0], t[3] - t[1] for t in trajectory_py]

    result = dict()
    result['res'] = trajectory_py
    result['type'] = 'rect'
    if duration > 0.0:
        result['fps'] = round(len(frames) / duration, 3)
    else:
        result['fps'] = 0.0
    return result


if __name__ == '__main__':

    args = parse_args()
    
    # setup tracker configuration
    cfg_path = args.config_file
    merge_cfg_from_file(cfg_path)
    if len(args.opts) > 0:
        merge_cfg_from_list(args.opts)
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    overwrite = args.overwrite > 0

    output_dir = os.path.join('output', 'LaSOT', cfg.TRACKER_TYPE, cfg.NAME)
    mkdir_if_not_exists(output_dir)

    dir_list = lasot.load_avaiable_dir_list()
    
    for ix, dir_name in enumerate(dir_list):
        base_name = os.path.basename(dir_name)
        output_path = os.path.join(output_dir, '{}.txt'.format(base_name))
        if os.path.exists(output_path) and not overwrite:
            print("[LaSOT ({}/{})] '{}' detected. Skip!".format(ix+1, len(dir_list), base_name))
            continue
        full_img_list, gt_rects = lasot.load_video_information(dir_name)
        seq = dict(
            frames=full_img_list,
            init_rect=gt_rects[0].tolist()
        )
        result = run_SiamTracker(seq, device)
        lasot.write_result(output_path, result['res'])
        print("[LaSOT ({}/{})] '{}' finished in {} FPS".format(ix+1, len(dir_list), base_name, result['fps']))
