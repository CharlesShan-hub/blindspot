#!/bin/bash

# Run Script
cd "$(dirname "$0")/.."
uv run -m scripts.source_curve \
    --dataset "/Volumes/Charles/data/blindpoint/source"
