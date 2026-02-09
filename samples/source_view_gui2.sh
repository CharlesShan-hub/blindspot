#!/bin/bash

# Run Script
cd "$(dirname "$0")/.."
uv run -m scripts.source_view_gui2 \
    --base_src "/Volumes/Charles/data/blindpoint/source" \
    --save_dir "/Volumes/Charles/data/blindpoint/source/temp" \
    --r 3 \
    --double_temp True \
    --window_width 1200 \
    --window_height 820
