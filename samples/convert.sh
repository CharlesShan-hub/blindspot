#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/convert.py"

python $PYTHON_SCRIPT \
    --src /Users/kimshan/Public/data/test \
    --dest /Users/kimshan/Public/data/blindpoint \
    --only_csv True