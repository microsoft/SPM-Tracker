from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def d_to_f(x):
    return list(map(lambda o:round(float(o),4), x))

def matlab_double_to_py_float(double):
    return list(map(d_to_f, double))

def ssd(x, y):
    if len(x) != len(y):
        sys.exit("cannot calculate ssd")
    s = 0
    for i in range(len(x)):
        s += (x[i] - y[i])**2
    return math.sqrt(s)


import math

from ..config import *
from .seq_config import *
from .eval_results import *
from .load_results import *
from .shift_bbox import *
from .split_seq import *
from .calc_seq_err_robust import *
from .calc_rect_center import *
