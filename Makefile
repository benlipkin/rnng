SHELL := /usr/bin/env bash
EXEC = python=3.6
PACKAGE = rnng
INSTALL = pip install -e .
ACTIVATE = source activate $(PACKAGE)
.DEFAULT_GOAL := help

## help      : print available build commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## update    : update repo with latest version from GitHub.
.PHONY : update
update :
	@git pull origin main

## env       : setup environment and install dependencies.
.PHONY : env
env : $(PACKAGE).egg-info/
$(PACKAGE).egg-info/ : setup.py requirements.txt
	@conda create -yn $(PACKAGE) $(EXEC)
	@$(ACTIVATE) ; $(INSTALL)

## setup     : download large files and prepare runtime.
.PHONY : setup
setup : env rnnlm_ptboanc_k_1024 rnng_td_ptboanc_n_1024
rnnlm_ptboanc_k_1024 : models/rnnlm_ptboanc_k_1024.pt
rnng_td_ptboanc_n_1024 : models/rnng_td_ptboanc_n_1024.pt
models/%.pt : setup/setup.sh
	@$(ACTIVATE) ; cd $(<D) ; bash $(<F) $(@F)

## test      : run testing suite.
.PHONY : test
test : env
	@$(ACTIVATE) ; 

## train      : train models from scratch.
.PHONY : train
train : 
	@echo "Training loop commented out. Use setup recipe to download pretrained models."
# train : env rnnlm_ptboanc_k rnng_td_ptboanc_n
# rnnlm_ptboanc_k : corpora_k models/rnnlm_ptboanc_k.pt
# rnng_td_ptboanc_n : corpora_n models/rnng_td_ptboanc_n.pt
# corpora_k : urnng/data/ptboanc-train.pkl
# corpora_n : rnng-pytorch/data/ptboanc-train.json
# models/%.pt : train/train.sh
# 	@$(ACTIVATE) ; cd $(<D) ; bash $(<F) $(@F)
# urnng/data/ptboanc-train.pkl : train/corpora.sh corpora/ptboanc_train.txt
# 	@$(ACTIVATE) ; cd $(<D) ; bash $(<F) $(@D)
# rnng-pytorch/data/ptboanc-train.json : train/corpora.sh corpora/ptboanc_train.txt
# 	@$(ACTIVATE) ; cd $(<D) ; bash $(<F) $(@D)