#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/source_detect.py"

python $PYTHON_SCRIPT \
    --dataset "/Users/kimshan/Public/data/blindpoint/source" \
    --method "curved_surface" \
    --result "/Users/kimshan/Public/library/blindspot/data/s6"