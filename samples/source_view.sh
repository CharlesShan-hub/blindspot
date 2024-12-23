#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/source_view.py"

python $PYTHON_SCRIPT \
    --dataset "/Users/kimshan/Public/data/blindpoint/source" \
    --index 19 \
    --method_list "origin|gb|curve3|curve4|curve5" \
    --need_save False \
    --save_path "/Users/kimshan/Public/data/blindpoint/tmp"