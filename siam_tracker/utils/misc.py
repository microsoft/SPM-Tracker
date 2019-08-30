# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os


def mkdir_if_not_exists(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

