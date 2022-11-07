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
    if [[ "$MODEL" == "rnnlm_ptboanc_k.pt" ]]; then
        python train_lm.py \
        --w_dim 512 \
        --h_dim 512 \
        --train_file data/ptboanc-train.pkl \
        --val_file data/ptboanc-val.pkl \
        --save_path "$MODEL_DIR/$MODEL"
    elif [[ "$MODEL" == "rnng_td_ptboanc_k.pt" ]]; then
        python train.py \
        --w_dim 512 \
        --h_dim 512 \
        --train_file data/ptboanc-train.pkl \
        --val_file data/ptboanc-val.pkl \
        --save_path "$MODEL_DIR/$MODEL" \
        --mode supervised
    else
        unk_model
    fi
elif [[ "$MODEL" =~ _n.pt ]]; then
    cd "$BASE_DIR/rnng-pytorch"
    if [[ "$MODEL" == "rnng_td_ptboanc_n.pt" ]]; then
        python train.py \
        --w_dim 512 \
        --h_dim 512 \
        --train_file data/ptboanc-train.json \
        --val_file data/ptboanc-val.json \
        --save_path "$MODEL_DIR/$MODEL" \
        --batch_size 128 \
        --fixed_stack \
        --strategy top_down \
        --optimizer adam
    elif [[ "$MODEL" == "rnng_lc_ptboanc_n.pt" ]]; then
        python train.py \
        --w_dim 512 \
        --h_dim 512 \
        --train_file data/ptboanc-train.json \
        --val_file data/ptboanc-val.json \
        --save_path "$MODEL_DIR/$MODEL" \
        --batch_size 128 \
        --fixed_stack \
        --strategy in_order \
        --optimizer adam
    else
        unk_model
    fi
else
    unk_model
fi
