#!/bin/bash

# Run Script
cd "$(dirname "$0")/.."
uv run -m scripts.source_compensation \
    --dataset "/Volumes/Charles/data/blindpoint/source" \
    --index 7 \
    --save_path "./assets/compensation_test.png" \
    --detection "gb" \
    --compensation "sobel" \
    --img_path "/Users/charles/workspace/data/blindlama/dataset/train/img/996.png" \
    --x1 65 --x2 100 --y1 150 --y2 190