#!/bin/bash

# Run Script
cd "$(dirname "$0")/.."
uv run -m scripts.source_view \
    --dataset "/Volumes/Charles/data/blindpoint/source" \
    --index 19 \
    --method_list "origin|gb|curve3|curve4|curve5" \
    --need_save False \
    --save_path "/Volumes/Charles/data/blindpoint/tmp"
