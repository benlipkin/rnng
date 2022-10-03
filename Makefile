SHELL := /usr/bin/env bash
EXEC = python=3.6
PACKAGE = urnng
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
setup : env $(PACKAGE)/models/rnng_ptb.pt
$(PACKAGE)/models/rnng_ptb.pt : setup/setup.sh
	@$(ACTIVATE) ; cd $(<D) ; bash $(<F)

## test      : run testing suite.
.PHONY : test
test : env
	@$(ACTIVATE) ; 

## train      : train models from scratch.
.PHONY : train
train : 
	@echo "Training loop commented out. Use setup recipe to download pretrained models."
# train : env rnng rnnlm
# rnng : corpora $(PACKAGE)/models/rnng_ptb.pt
# rnnlm : corpora $(PACKAGE)/models/rnnlm_ptb.pt
# corpora : $(PACKAGE)/data/ptb-train.pkl
# $(PACKAGE)/models/rnng_ptb.pt : $(PACKAGE)/train.py
# 	@$(ACTIVATE) ; cd train ; bash train_rnng.sh
# $(PACKAGE)/models/rnnlm_ptb.pt : $(PACKAGE)/train_lm.py
# 	@$(ACTIVATE) ; cd train ; bash train_rnnlm.sh
# $(PACKAGE)/data/ptb-train.pkl : $(PACKAGE)/data/ptb_train.txt
# 	@$(ACTIVATE) ; cd train ; bash prep_corpora.sh