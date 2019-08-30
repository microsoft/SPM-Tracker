# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='crop_and_resize',
    ext_modules=[
        CUDAExtension('crop_and_resize', [
            'src/crop_and_resize.cpp',
            'src/crop_and_resize_cpu.cpp',
            'src/crop_and_resize_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})
