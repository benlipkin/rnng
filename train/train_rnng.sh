#!/usr/bin/env bash

set -e
cd ../urnng

# Train RNNG on PTB
python train.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path models/rnng_ptb.pt --mode supervised
