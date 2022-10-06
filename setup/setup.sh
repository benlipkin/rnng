#!/usr/bin/env bash

set -e

conda install -yc anaconda wget

HOME_DIR="$(dirname $(pwd))"
FILEID="$1"
FNAME="$HOME_DIR/models/$FILEID"

BASE="https://huggingface.co/datasets/benlipkin/rnng-brainscore/resolve/main"
if [ ! -f "$FNAME" ]; then
    mkdir -p "$(dirname $FNAME)"
    wget -O "$FNAME" "$BASE/$FILEID"
fi
