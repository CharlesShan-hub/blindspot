#!/bin/bash

# Run Script
cd "$(dirname "$0")/.."
uv run -m scripts.source_detect \
    --dataset "/Volumes/Charles/data/blindpoint/source" \
    --index 0 \
    --method "gb" \
    --curve_sigma 6 \
    --need_save True \
    --result "/Volumes/Charles/data/blindpoint/source/bad/gb"

# gb
# curved_surface