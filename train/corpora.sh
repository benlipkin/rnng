#!/usr/bin/env bash

set -e

BASE_DIR="../$(dirname $1)"
cd "$BASE_DIR"

if [ ! -f "data/ptboanc.vocab" ]; then
    python preprocess.py --trainfile "../corpora/ptboanc_train.txt" --valfile "../corpora/ptboanc_valid.txt" --testfile "../corpora/ptboanc_test.txt" --outputfile "data/ptboanc" --vocabminfreq 1
fi