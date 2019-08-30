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

from siam_tracker.benchmarks.otb import butil
from siam_tracker.benchmarks.otb import Result
import siam_tracker.benchmarks.otb.config as otb_cfg

from siam_tracker.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list
from siam_tracker.core.inference import SiamTracker

import siam_tracker.utils.boxes as ubox
from siam_tracker.utils.misc import mkdir_if_not_exists

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
logging.getLogger().setLevel(logging.INFO)

BENCHMARK_SETS = ['tb100', 'tb50', 'tc']


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
    parser.add_argument('--seq', dest='seq',
                        help='sequences to evaluate',
                        default='tb100',
                        type=str)
    parser.add_argument('--eval', dest='eval',
                        help='evaluation type',
                        default='OPE',
                        type=str)

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

    tic = time.time()
    frames = seq.s_frames
    init_rect = seq.init_rect
    x, y, width, height = init_rect  # OTB format
    init_rect[0] -= 1
    init_rect[1] -= 1
    init_bb = list(ubox.xywh_to_xyxy(init_rect))
    trajectory_py = tracker.track(init_box=init_bb, input_str=frames, box_format='tlbr')
    duration = time.time() - tic

    trajectory = [Rectangle(t[0] + 1., t[1] + 1., t[2] - t[0], t[3] - t[1]) for t in trajectory_py]

    result = dict()
    result['res'] = trajectory
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
    return result


def run_trackers(seqs, evalType, device):
    numSeq = len(seqs)

    tracker_results = []
    
    tracker_dir = cfg.TRACKER_TYPE + '/' +  cfg.NAME
    
    for idxSeq in range(0, numSeq):
        s = seqs[idxSeq]
        
        subSeqs, subAnno = butil.get_sub_seqs(s, 20.0, evalType)

        if not otb_cfg.OVERWRITE_RESULT:
            trk_src = os.path.join(otb_cfg.RESULT_SRC.format(evalType), tracker_dir)
            result_src = os.path.join(trk_src, s.name+'.json')
            if os.path.exists(result_src):
                seqResults = butil.load_seq_result(evalType, tracker_dir, s.name)
                tracker_results.append(seqResults)
                continue

        seqResults = []
        seqLen = len(subSeqs)
        for idx in range(seqLen):
            logging.info('{0}_{1}, {2}_{3}:{4}/{5} - {6}'.format(
                'Name', tracker_dir, idxSeq + 1, s.name, idx + 1, seqLen, \
                evalType))
            
            subS = subSeqs[idx]
            subS.name = s.name + '_' + str(idx)

            res = run_SiamTracker(subS, device)

            if evalType == 'SRE':
                r = Result(tracker_dir, s.name, subS.startFrame, subS.endFrame,
                    res['type'], evalType, res['res'], res['fps'], otb_cfg.shiftTypeSet[idx])
            else:
                r = Result(tracker_dir, s.name, subS.startFrame, subS.endFrame,
                    res['type'], evalType, res['res'], res['fps'], None)
            try: r.tmplsize = butil.d_to_f(res['tmplsize'][0])
            except: pass
            r.refresh_dict()
            seqResults.append(r)
        #end for subseqs
        if otb_cfg.SAVE_RESULT:
            butil.save_seq_result(seqResults)
        tracker_results.append(seqResults)

    return tracker_results


def print_results(seqs, results, eval_type):

    tracker_dir = cfg.TRACKER_TYPE + '/' +  cfg.NAME

    if len(results) > 0:
        evalResults, attrList = butil.calc_result(tracker_dir, seqs, results, eval_type)
        all_err = []
        for seq in seqs:
            n_success_frm = 0
            n_fail_frm = 0
            for ec in seq.errCenter:
                if ec < 20.0:
                    n_success_frm += 1
                else:
                    n_fail_frm += 1
            err = n_success_frm/float(n_success_frm+n_fail_frm)
            all_err.append(err)
        logging.info("Result of Sequences\t -- '{0}'".format(tracker_dir))
        for i_seq, seq in enumerate(seqs):
            try:
                logging.info('\t\'{0}\'{1}'.format(seq.name, " "*(12 - len(seq.name))) 
                    + "\taveCoverage : {0:.3f}".format(sum(seq.aveCoverage)/len(seq.aveCoverage)) 
                    + "\taveErrCenter : {0:.3f}".format(sum(seq.aveErrCenter)/len(seq.aveErrCenter))
                    + "\tprec@20 : {0:.3f}".format(all_err[i_seq]))
            except:
                logging.error('\t\'{0}\'  ERROR!!'.format(seq.name))
        avg_err = sum(all_err) / len(all_err)
        logging.info('Precision plot score for ALL: {}'.format(round(avg_err,3)))
        logging.info("Result of attributes\t -- '{0}'".format(tracker_dir))
        for attr in attrList:
            logging.info("\t\'{0}\'".format(attr.name)
                + "\toverlap : {0:02.1f}%".format(attr.overlap)
                + "\tfailures : {0:.1f}".format(attr.error)
                + "\tAUC : {0:.3f}".format(sum(attr.successRateList) / len(attr.successRateList)))
        if otb_cfg.SAVE_RESULT : 
            butil.save_scores(attrList, tracker_dir)


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
    
    # setup benchmark sequences
    seqs = []
    if args.seq not in BENCHMARK_SETS:
        loadSeqs = [x.strip() for x in args.seq.split(',')]
    else:
        loadSeqs = args.seq
    eval_type = args.eval
    if otb_cfg.SETUP_SEQ:
        logging.info('Setup sequences...')
        butil.setup_seqs(loadSeqs)
    logging.info('Starting benchmark for {0} trackers, evalTypes : {1}'.format(
        cfg.NAME, eval_type))
    
    seqNames = butil.get_seq_names(loadSeqs)
    seqs = butil.load_seq_configs(seqNames)

    results = run_trackers(seqs, eval_type, device)
    
    print_results(seqs, results, eval_type)
    