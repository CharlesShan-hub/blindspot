#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/view.py"

python $PYTHON_SCRIPT \
    --base_src "/Users/kimshan/Public/data/blindpoint" \
    --save_dir "/Users/kimshan/Public/data/blindpoint/temp" \
    --r 3 \
    --double_temp True \
    --window_width 1200 \
    --window_height 820