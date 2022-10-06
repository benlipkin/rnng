#!/usr/bin/env bash

set -e

HOME_DIR="$(dirname $(pwd))"

wget=$(conda list wget | grep -c wget)
if [ "$wget" -eq 0 ]; then
    conda install -yc anaconda wget
fi

FILEID="$1"
FNAME="$HOME_DIR/models/$FILEID"

BASE="https://huggingface.co/datasets/benlipkin/rnng-brainscore/resolve/main"
if [ ! -f "$FNAME" ]; then
    wget -O "$FNAME" "$BASE/$FILEID"
fi
