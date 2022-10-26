#!/usr/bin/env python3
import argparse
import json
import sys
from tkinter import E

import numpy as np
import torch
from torch import cuda

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--measure", type=str, required=True)
parser.add_argument("--context", type=str, required=True)
parser.add_argument("--text", type=str, required=True)
parser.add_argument("--gpu", default=-1, type=int, help="which gpu to use")
parser.add_argument("--seed", default=3435, type=int, help="random seed")
parser.add_argument("--include_eos", default=False, type=bool)
parser.add_argument("--sentence_only", action="store_true")


def load_model(args):
    cuda.set_device(args.gpu) if args.gpu != -1 else None
    checkpoint = torch.load(args.model)
    word2idx = checkpoint["word2idx"]
    idx2word = checkpoint["idx2word"]
    model = checkpoint["model"]
    model.eval()
    model.cuda() if args.gpu != -1 else None
    return model, word2idx, idx2word


def prep_tokens(text, word2idx, include_eos=False):
    def prep_token(token):
        return token if token in word2idx else "<unk>"

    text = ["<s>"] + text.strip().split() + ["</s>"]
    tokens = torch.Tensor([word2idx[prep_token(w)] for w in text]).unsqueeze(0).long()
    if not include_eos:
        tokens = tokens[:, :-1]
    return tokens


def get_measure(model, tokens, lastn, idx2word, args):
    tokens = tokens.cuda() if args.gpu != -1 else tokens
    with torch.no_grad():
        measure = model.get_measure(tokens, args.measure, lastn)
    if args.measure == "next-word":
        measure = idx2word[measure]
    else:
        measure = measure.tolist()
    if "token" in args.measure:
        return {
            "tokens": tokens.cpu().numpy().squeeze()[-lastn:].tolist(),
            "measure": measure,
        }
    else:
        return {"measure": measure}


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, word2idx, idx2word = load_model(args)
    text = args.text if args.sentence_only else args.context
    tokens = prep_tokens(text, word2idx, args.include_eos)
    lastn = len(args.text.strip().split())
    measure = get_measure(model, tokens, lastn, idx2word, args)
    response_json = json.dumps(measure)
    sys.stdout.write(response_json)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
