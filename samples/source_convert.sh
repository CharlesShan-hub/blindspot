#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/source_convert.py"

python $PYTHON_SCRIPT \
    --src /Users/kimshan/Public/data/test \
    --dest /Users/kimshan/Public/data/blindpoint/source \
    --only_csv True