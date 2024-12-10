#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/detect.py"

python $PYTHON_SCRIPT \
    --dataset "/Users/kimshan/Public/data/blindpoint" \
    --method "curved_surface" \
    --result "/Users/kimshan/Public/library/blindspot/data/s6"