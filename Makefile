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
setup : env
	@$(ACTIVATE) ; 

## test      : run testing suite.
.PHONY : test
test : env
	@$(ACTIVATE) ; 