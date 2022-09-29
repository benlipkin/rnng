#!/usr/bin/env bash

set -e
cd ../urnng

# Train RNNLM on PTB
python train_lm.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path models/rnnlm_ptb.pt