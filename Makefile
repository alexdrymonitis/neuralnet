# Makefile for neural

lib.name = neuralnet

class.sources = neuralnet.c

datafiles = neuralnet-help.pd README.md

PDLIBBUILDER_DIR=../pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder
