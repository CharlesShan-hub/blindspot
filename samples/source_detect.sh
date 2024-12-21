#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/source_detect.py"

python $PYTHON_SCRIPT \
    --dataset "/Users/kimshan/Public/data/blindpoint/source" \
    --index 0 \
    --method "gb" \
    --curve_sigma 6 \
    --need_save True \
    --result "/Users/kimshan/Public/data/blindpoint/source/bad/gb"

# gb
# curved_surface