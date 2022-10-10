import argparse
import pathlib
import subprocess
import sys

import nltk

parser = argparse.ArgumentParser()

parser.add_argument("--text", type=str, required=True, help="string to be evaluated.")
parser.add_argument("--model", type=str, required=True, help="evaluation model.")
parser.add_argument("--rep", type=str, required=True, help="representation to extract.")
parser.add_argument("--gpu", default=-1, type=int, help="which gpu to use. -1 for cpu.")

reps = {
    "rnnlm_ptb_k": ["rnn.lm.emb.mean", "rnn.lm.lstm.mean", "rnn.lm.lstm.last"],
    "rnng_td_ptb_n": [],
    "rnng_lc_ptb_n": [],
}


def tokenize(text):
    tokens = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize(
        text, convert_parentheses=True
    )
    return " ".join(tokens)


def get_path_to_model(model):
    model_dir = pathlib.Path(__file__).parents[1] / "models"
    path = model_dir / f"{model}.pt"
    return path


def get_rep_json(text, model, rep, gpu):
    src_dir = "urnng" if "rnnlm" in model.stem else "rnng-pytorch"
    src_file = f"{model.stem.split('_')[0]}_brainscore"
    cmd = ["cd", str(src_dir), ";"]
    cmd += ["python", "-m", str(src_file)]
    cmd += ["--text", f'"{str(text)}"']
    cmd += ["--model", str(model)]
    cmd += ["--rep", str(rep)]
    cmd += ["--gpu", str(gpu)]
    output = subprocess.check_output(" ".join(cmd), shell=True)
    rep_json = output.decode("utf-8")
    return rep_json


def main(args):
    if not args.model in reps.keys():
        raise ValueError(f"model must be one of {reps.keys()}.")
    if not args.rep in reps[args.model]:
        raise ValueError(f"representation must be one of {reps[args.model]}.")
    text = tokenize(args.text)
    model = get_path_to_model(args.model)
    rep_json = get_rep_json(text, model, args.rep, args.gpu)
    sys.stdout.write(rep_json)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
