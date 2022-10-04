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
	@$(ACTIVATE) ;
# setup : env models/rnnlm_ptb_k.pt
# models/rnnlm_ptb_k.pt : setup/setup.sh
# 	@$(ACTIVATE) ; cd $(<D) ; bash $(<F)

## test      : run testing suite.
.PHONY : test
test : env
	@$(ACTIVATE) ; 

## train      : train models from scratch.
.PHONY : train
# train : 
# 	@echo "Training loop commented out. Use setup recipe to download pretrained models."
train : env rnnlm_ptb_k rnng_td_ptb_k rnng_td_ptb_n rnng_lc_ptb_n
rnnlm_ptb_k : corpora_k models/rnnlm_ptb_k.pt
rnng_td_ptb_k : corpora_k models/rnng_td_ptb_k.pt
rnng_td_ptb_n : corpora_n models/rnng_td_ptb_n.pt
rnng_lc_ptb_n : corpora_n models/rnng_lc_ptb_n.pt
corpora_k : urnng/data/ptb-train.pkl
corpora_n : rnng-pytorch/data/ptb-train.json
models/%.pt : train/train.sh
	@$(ACTIVATE) ; cd $(<D) ; bash $(<F) $(@F)
corpora := $(wildcard */data/ptb-train.*)
$(corpora) : train/corpora.sh corpora/ptb_train.txt
	@$(ACTIVATE) ; cd $(<D) ; bash $(<F) $(@D)