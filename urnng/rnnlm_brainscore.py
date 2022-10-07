#!/usr/bin/env python3
import argparse
import json
import sys

import numpy as np
import torch
from torch import cuda

parser = argparse.ArgumentParser()

parser.add_argument("--text_path", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--rep", type=str, required=True)
parser.add_argument("--include_eos", default=False, type=bool)
parser.add_argument("--gpu", default=-1, type=int, help="which gpu to use")
parser.add_argument("--seed", default=3435, type=int, help="random seed")


def load_model(args):
    cuda.set_device(args.gpu) if args.gpu != -1 else None
    checkpoint = torch.load(args.model_path)
    word2idx = checkpoint["word2idx"]
    model = checkpoint["model"]
    model.eval()
    model.cuda() if args.gpu != -1 else None
    return model, word2idx


def load_sent(args):
    with open(args.text_path, "r") as f:
        sent = f.read().strip().split()
    return sent


def prep_sent(sent, word2idx, include_eos=False):
    def prep_token(token):
        return token if token in word2idx else "<unk>"

    sent = ["<s>"] + sent + ["</s>"]
    sent = torch.Tensor([word2idx[prep_token(w)] for w in sent]).unsqueeze(0).long()
    if not include_eos:
        sent = sent[:, :-1]
    return sent


def get_rep(sent, model, args):
    sent = sent.cuda() if args.gpu != -1 else sent
    with torch.no_grad():
        if any([layer in args.rep for layer in ["emb", "lstm"]]):
            rep = model.get_layer_rep(sent, args.rep)
            assert rep.shape[0] == 1
        else:
            raise ValueError("Invalid rep argument")
    return rep.cpu().numpy().squeeze()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, word2idx = load_model(args)
    sent = prep_sent(load_sent(args), word2idx, args.include_eos)
    rep = get_rep(sent, model, args)
    rep_json = json.dumps(rep.tolist())
    sys.stdout.write(rep_json)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
