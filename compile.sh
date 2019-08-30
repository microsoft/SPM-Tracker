#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building crop_and_resize op..."
cd siam_tracker/ops/crop_and_resize
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace


echo "Building bbox opertations"
cd ../..
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
