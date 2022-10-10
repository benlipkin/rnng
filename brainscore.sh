#!/usr/bin/env bash

set -e

source activate rnng

python -m brainscore --text "$1" --model "$2" --rep "$3"