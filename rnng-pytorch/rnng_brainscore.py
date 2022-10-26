#!/usr/bin/env python3
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--measure", type=str, required=True)
parser.add_argument("--context", type=str, required=True)
parser.add_argument("--text", type=str, required=True)
parser.add_argument("--gpu", default=-1, type=int, help="which gpu to use")
parser.add_argument("--seed", default=3435, type=int, help="random seed")
parser.add_argument("--include_eos", default=False, type=bool)
parser.add_argument("--sentence_only", action="store_true")


def evaluate_model(text, model, measure):
    cmd = f"python -m beam_search"
    cmd += f' --text "{text}" --model_file {model}'
    cmd += f" --measure {measure} --device cpu"
    output = subprocess.check_output(cmd, shell=True)
    response_json = output.decode("utf-8")
    return response_json


def main(args):
    assert args.sentence_only, "Only sentence-level evaluation is supported."
    response_json = evaluate_model(args.text, args.model, args.measure)
    sys.stdout.write(response_json)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
