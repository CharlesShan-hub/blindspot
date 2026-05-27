#!/bin/bash

# Run Script
cd "$(dirname "$0")/.."
uv run -m scripts.source_compare_curve \
    --dataset "/Volumes/Charles/data/blindpoint/source" \
    --index 35 \
    --save_path "./curve_compare.png"