from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

import numpy as np

_NP_INCLUDE_DIRS = np.get_include()

# Extension modules
ext_modules = [
    Extension(
        name='utils.boxes.cython_bbox',
        sources=[
            'utils/boxes/cython_bbox.pyx'
        ],
        extra_compile_args=[
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    ),
    Extension(
        name='utils.boxes.cython_nms',
        sources=[
            'utils/boxes/cython_nms.pyx'
        ],
        extra_compile_args=[
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    )]

setup(
    name='SPMTracker',
    ext_modules=cythonize(ext_modules)
)
