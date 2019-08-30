from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import sys
import os
import os.path as osp

# sys.path.append(os.getcwd())

OTB_DATA_DIR = 'data/benchmarks/otb/data/'

LOG_DIR = 'output'

############### benchmark config ####################
OTB_DIR = osp.dirname(__file__)

WORKDIR = os.path.abspath('.')

SEQ_SRC = OTB_DATA_DIR

RESULT_SRC = osp.join(LOG_DIR, 'OTB', 'results/{0}/') # '{0} : OPE, SRE, TRE'

SETUP_SEQ = True

SAVE_RESULT = True

OVERWRITE_RESULT = True

SAVE_IMAGE = False

USE_INIT_OMIT = True

# sequence configs
DOWNLOAD_SEQS = False
DOWNLOAD_URL = "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq_new/{0}.zip"
ATTR_LIST_FILE = 'attr_list.txt'
ATTR_DESC_FILE = 'attr_desc.txt'
TB_50_FILE = 'tb_50.txt'
TB_100_FILE = 'tb_100.txt'
TC_FILE = 'tc.txt'
CVPR_13_FILE = 'cvpr13.txt' 
ATTR_FILE = 'attrs.txt'
INIT_OMIT_FILE = 'init_omit.txt'
GT_FILE = 'groundtruth_rect.txt'

shiftTypeSet = ['left','right','up','down','topLeft','topRight',
        'bottomLeft', 'bottomRight','scale_8','scale_9','scale_11','scale_12']

# for evaluating results
thresholdSetOverlap = [x/float(20) for x in range(21)]
thresholdSetError = range(0, 51)

# for drawing plot
MAXIMUM_LINES = 10
LINE_COLORS = ['b','g','r','c','m','y','k', '#880015', '#FF7F27', '#00A2E8']
