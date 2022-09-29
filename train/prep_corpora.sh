#!/usr/bin/env bash

set -e
cd ../urnng

# Prep PTB
python preprocess.py --trainfile data/ptb_train.txt --valfile data/ptb_valid.txt --testfile data/ptb_test.txt --outputfile data/ptb