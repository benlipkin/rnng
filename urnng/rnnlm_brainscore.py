#!/usr/bin/env python3
import argparse
import json
import sys

import numpy as np
import torch
from torch import cuda

parser = argparse.ArgumentParser()

parser.add_argument("--text", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--rep", type=str, required=True)
parser.add_argument("--gpu", default=-1, type=int, help="which gpu to use")
parser.add_argument("--seed", default=3435, type=int, help="random seed")
parser.add_argument("--include_eos", default=False, type=bool)


def load_model(args):
    cuda.set_device(args.gpu) if args.gpu != -1 else None
    checkpoint = torch.load(args.model)
    word2idx = checkpoint["word2idx"]
    model = checkpoint["model"]
    model.eval()
    model.cuda() if args.gpu != -1 else None
    return model, word2idx


def prep_tokens(text, word2idx, include_eos=False):
    def prep_token(token):
        return token if token in word2idx else "<unk>"

    text = ["<s>"] + text.strip().split() + ["</s>"]
    tokens = torch.Tensor([word2idx[prep_token(w)] for w in text]).unsqueeze(0).long()
    if not include_eos:
        tokens = tokens[:, :-1]
    return tokens


def get_rep(tokens, model, args):
    tokens = tokens.cuda() if args.gpu != -1 else tokens
    with torch.no_grad():
        rep = model.get_rep(tokens, args.rep)
        assert rep.shape[0] == 1
    return rep.cpu().numpy().squeeze()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, word2idx = load_model(args)
    tokens = prep_tokens(args.text, word2idx, args.include_eos)
    rep = get_rep(tokens, model, args)
    rep_json = json.dumps(rep.tolist())
    sys.stdout.write(rep_json)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
