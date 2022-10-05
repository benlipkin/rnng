#!/usr/bin/env bash

set -e

BASE_DIR="../$(dirname $1)"
cd "$BASE_DIR"

if [ ! -f "data/ptb.vocab" ]; then
    python preprocess.py --trainfile "../corpora/ptb_train.txt" --valfile "../corpora/ptb_valid.txt" --testfile "../corpora/ptb_test.txt" --outputfile "data/ptb" --vocabminfreq 1
fi
