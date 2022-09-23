#!/usr/bin/env bash

set -e
cd ../urnng

# Test preprocessing
python preprocess.py --trainfile data/train.txt --valfile data/valid.txt --testfile data/test.txt --outputfile data/ptb --vocabminfreq 1 --lowercase 0 --replace_num 0 --batchsize 16

# Test (U)RNNG training
python train.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path models/urnng.pt --mode unsupervised --gpu 0 --num_epochs 1
python train.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path models/rnng.pt --mode supervised --gpu 0 --num_epochs 1

# Test RNNLM training
python train_lm.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --test_file data/ptb-test.pkl --save_path models/lm.pt --gpu 0 --num_epochs 1

# Test (U)RNNG perplexity evaluation
python eval_ppl.py --model_file models/urnng.pt --test_file data/ptb-test.pkl --samples 100 --gpu 0
python eval_ppl.py --model_file models/rnng.pt --test_file data/ptb-test.pkl --samples 100 --gpu 0

# Test RNNLM perplexity evaluation
python train_lm.py --train_from models/lm.pt --test_file data/ptb-test.pkl --test 1 --gpu 0

echo "Test runs completed successfully."
