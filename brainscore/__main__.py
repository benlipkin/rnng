import argparse
import pathlib
import subprocess
import sys

import nltk

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, help="evaluation model.")
parser.add_argument("--measure", type=str, required=True, help="measure to extract.")
parser.add_argument("--context", type=str, required=True, help="text context.")
parser.add_argument("--text", type=str, required=True, help="string to be evaluated.")
parser.add_argument("--gpu", default=-1, type=int, help="which gpu to use. -1 for cpu.")

supported = {
    "rnn-lm-ptb": {
        "path": pathlib.Path(__file__).parents[1] / "models" / "rnnlm_ptb_k.pt",
        "measures": ["next-word", "token-logits", "emb-mean", "lstm-mean", "lstm-last"],
    },
    "rnn-tdg-ptb": {
        "path": pathlib.Path(__file__).parents[1] / "models" / "rnng_td_ptb_n.pt",
        "measures": [],
    },
    "rnn-lcg-ptb": {
        "path": pathlib.Path(__file__).parents[1] / "models" / "rnng_lc_ptb_n.pt",
        "measures": [],
    },
}


def tokenize(text):
    tokens = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize(
        text, convert_parentheses=True
    )
    return " ".join(tokens)


def get_response_json(model, measure, context, text, gpu):
    src_dir = "urnng" if "lm" in model.stem else "rnng-pytorch"
    src_file = "rnnlm_brainscore" if "lm" in model.stem else "rnng_brainscore"
    cmd = ["cd", str(src_dir), "&&"]
    cmd += ["python", "-m", str(src_file)]
    cmd += ["--model", str(model)]
    cmd += ["--measure", str(measure)]
    cmd += ["--context", f'"{str(context)}"']
    cmd += ["--text", f'"{str(text)}"']
    cmd += ["--gpu", str(gpu)]
    output = subprocess.check_output(" ".join(cmd), shell=True)
    rep_json = output.decode("utf-8")
    return rep_json


def main(args):
    if not args.model in supported.keys():
        raise ValueError(f"model must be one of {supported.keys()}.")
    if not args.measure in supported[args.model]["measures"]:
        raise ValueError(f"measure must be one of {supported[args.model]['measures']}.")
    context = tokenize(args.context)
    text = tokenize(args.text)
    if not context[-len(text) :] == text:
        raise ValueError(
            "context must end with text, e.g., context='I am a wug', text='a wug'"
        )
    model = supported[args.model]["path"]
    response_json = get_response_json(model, args.measure, context, text, args.gpu)
    sys.stdout.write(response_json)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
