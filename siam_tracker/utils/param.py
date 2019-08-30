# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from collections import OrderedDict


def load_weights(model, model_path, verbose=False):
    st = model.state_dict()
    load_st = OrderedDict()
    weights = torch.load(model_path)
    for k, v in st.items():
        if k in weights:
            if verbose:
                print('Loading {}'.format(k))
            load_st[k] = weights[k]
        else:
            if verbose:
                print('Randomly init {}'.format(k))
            load_st[k] = st[k]
    model.load_state_dict(load_st)

