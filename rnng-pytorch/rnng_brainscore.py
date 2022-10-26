#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from hashlib import md5
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--measure", type=str, required=True)
parser.add_argument("--context", type=str, required=True)
parser.add_argument("--text", type=str, required=True)
parser.add_argument("--gpu", default=-1, type=int, help="which gpu to use")
parser.add_argument("--seed", default=3435, type=int, help="random seed")
parser.add_argument("--include_eos", default=False, type=bool)
parser.add_argument("--sentence_only", action="store_true")


def write_temp_file(context):
    fname = (
        Path(__file__).parents[1]
        / "evaluation"
        / f"{md5(context.encode()).hexdigest()}.tokens"
    )
    fname.parent.mkdir(exist_ok=True, parents=True)
    with open(fname, "w") as f:
        f.write(context)
    return fname


def evaluate_model(text_file, model_file):
    model_name = Path(model_file).stem
    text_file_stem = text_file.parent / text_file.stem
    surprisal_file = text_file_stem.with_suffix(".surprisal")
    parsed_file = text_file_stem.with_suffix(".parsed")
    rep_file = text_file_stem.with_suffix(f".{model_name}.reps")
    cmd = f"python -m beam_search"
    cmd += f" --test_file {text_file} --model_file {model_file}"
    cmd += f" --rep_file {rep_file} --device cpu"
    cmd += f" --lm_output_file {surprisal_file} > {parsed_file}"
    subprocess.run(cmd, shell=True, check=True)
    return surprisal_file, parsed_file, rep_file


def load_reps(rep_file):
    with open(rep_file) as f:
        reps = json.loads(f.read())
    return reps


def load_surprisal(surprisal_file):
    raise NotImplementedError("not yet implemented")


def prepare_response(rep_file, surprisal_file, args):
    reps = load_reps(rep_file)
    if args.measure in reps.keys():
        measure = reps[args.measure]
        return {"measure": measure}
    elif args.measure == "token-suprisal":
        surprisal = load_surprisal(surprisal_file)
        raise NotImplementedError("not yet implemented")
    else:
        raise RuntimeError(f"Unknown measure: {args.measure}")


def main(args):
    assert args.sentence_only, "Only sentence-level evaluation is supported."
    text_file = write_temp_file(args.text)  # only supports sentence-level inference
    surprisal_file, parsed_file, rep_file = evaluate_model(text_file, args.model)
    response = prepare_response(rep_file, surprisal_file, args)
    response_json = json.dumps(response)
    sys.stdout.write(response_json)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
