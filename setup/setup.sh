#!/usr/bin/env bash

set -e

wget=$(conda list wget | grep -c wget)
if [ "$wget" -eq 0 ]; then
    conda install -yc anaconda wget
fi

f="$1"
HOME_DIR="$(dirname $(pwd))"
if [[ "$f" =~ data ]]; then
    FNAME="$HOME_DIR/$(dirname $f)/data/ptb.vocab"
    if [[ "$f" =~ urnng ]]; then
        FILEID="ptb_k.vocab"
    else
        FILEID="ptb_n.vocab"
    fi
else
    FNAME="$HOME_DIR/models/$f"
    FILEID="$f"
fi

BASE="https://huggingface.co/datasets/benlipkin/rnng-brainscore/resolve/main"
if [ ! -f "$FNAME" ]; then
    wget -O "$FNAME" "$BASE/$FILEID"
fi
