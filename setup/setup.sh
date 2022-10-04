#!/usr/bin/env bash

set -e

conda install -yc anaconda wget

HOME_DIR="${1:-$(dirname $(pwd))}"
MODEL_DIR="$HOME_DIR/models"
cd "$MODEL_DIR"

for MODEL in {IN_PROGRESS}
do
    if [ ! -f "$MODEL" ]; then
        wget -O "$MODEL" "https://huggingface.co/datasets/benlipkin/rnng-brainscore/resolve/main/$MODEL"
    fi
done