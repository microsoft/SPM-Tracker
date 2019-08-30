# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .config import spm_tracker_cfgs
from .inference import SPMTrackerInference

info = dict(
    config=spm_tracker_cfgs,
    inference_wrapper=SPMTrackerInference
)