#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/source_view_gui.py"

python $PYTHON_SCRIPT \
    --base_src "/Users/kimshan/Public/data/blindpoint/source" \
    --save_dir "/Users/kimshan/Public/data/blindpoint/source/temp" \
    --r 3 \
    --double_temp True \
    --window_width 1200 \
    --window_height 820