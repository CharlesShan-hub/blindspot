#!/bin/bash

# Run Script
cd "$(dirname "$0")/.."
uv run -m scripts.source_convert \
    --src /Volumes/Charles/data/blindpoint/origin \
    --dest /Volumes/Charles/data/blindpoint/source \
    --skip_image True