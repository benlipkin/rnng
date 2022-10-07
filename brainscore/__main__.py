import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import nltk

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, required=True, help="string to be evaluated.")
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


def save_input_str_to_file(text):
    def md5(text):
        return hashlib.md5(text.encode()).hexdigest()

    input_dir = pathlib.Path(__file__).parents[1] / "inputs"
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"{md5(text)}.txt"
    with open(path, "w") as f:
        f.write(text)
    return path


def get_path_to_model(model):
    model_dir = pathlib.Path(__file__).parents[1] / "models"
    path = model_dir / f"{model}.pt"
    return path


def get_rep_json(text_path, model_path, rep, gpu):
    src_dir = "urnng" if "rnnlm" in model_path.stem else "rnng-pytorch"
    src_file = f"{model_path.stem.split('_')[0]}_brainscore"
    cmd = ["cd", str(src_dir), ";"]
    cmd += ["python", "-m", str(src_file)]
    cmd += ["--text_path", str(text_path)]
    cmd += ["--model_path", str(model_path)]
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
    tokenized = tokenize(args.input)
    text_path = save_input_str_to_file(tokenized)
    model_path = get_path_to_model(args.model)
    rep_json = get_rep_json(text_path, model_path, args.rep, args.gpu)
    sys.stdout.write(rep_json)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
