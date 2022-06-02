# Hisilicon Hi35xx sample Makefile
include ../Makefile.param

CFLAGS += -I$(PWD)/sample
CFLAGS += -I$(PWD)/encapsulation/include

SRCS := $(wildcard *.c)
SRCS += $(wildcard ./sample/*.c)
SRCS += $(wildcard ./encapsulation/src/*.c)
SRCS += $(wildcard $(PWD)/../common/*.c)
TARGET := sample_dsp_main

# compile linux or HuaweiLite
include $(PWD)/../../../$(OSTYPE).mak
