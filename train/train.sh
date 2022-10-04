#!/usr/bin/env bash

set -e

BASE_DIR="$(dirname $(pwd))"
MODEL_DIR="$BASE_DIR/models"
MODEL=$1

if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
fi

if [ -f "$MODEL_DIR/$MODEL" ]; then
    echo "$MODEL already trained."
    exit 0
fi

unk_model () {
    echo "$MODEL is an invalid model name."
    exit 1
}

if [[ "$MODEL" =~ _k.pt ]]; then
    cd "$BASE_DIR/urnng"
    if [[ "$MODEL" == "rnnlm_ptb_k.pt" ]]; then
        python train_lm.py \
        --train_file data/ptb-train.pkl \
        --val_file data/ptb-val.pkl \
        --save_path "$MODEL_DIR/$MODEL"
    elif [[ "$MODEL" == "rnng_td_ptb_k.pt" ]]; then
        python train.py \
        --train_file data/ptb-train.pkl \
        --val_file data/ptb-val.pkl \
        --save_path "$MODEL_DIR/$MODEL" \
        --mode supervised
    else
        unk_model
    fi
elif [[ "$MODEL" =~ _n.pt ]]; then
    cd "$BASE_DIR/rnng-pytorch"
    if [[ "$MODEL" == "rnng_td_ptb_n.pt" ]]; then
        python train.py \
        --train_file data/ptb-train.json \
        --val_file data/ptb-val.json \
        --save_path "$MODEL_DIR/$MODEL" \
        --fixed_stack \
        --strategy top_down \
        --optimizer adam
    elif [[ "$MODEL" == "rnng_lc_ptb_n.pt" ]]; then
        python train.py \
        --train_file data/ptb-train.json \
        --val_file data/ptb-val.json \
        --save_path "$MODEL_DIR/$MODEL" \
        --fixed_stack \
        --strategy in_order \
        --optimizer adam
    else
        unk_model
    fi
else
    unk_model
fi
