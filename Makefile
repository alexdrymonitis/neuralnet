# Makefile for neural

lib.name = neuralnet

class.sources = neuralnet.c

common.sources = dense.h dense.c

datafiles = neuralnet-help.pd README.md

PDLIBBUILDER_DIR=../pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder
