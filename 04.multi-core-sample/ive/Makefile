# Hisilicon Hi35xx sample Makefile

include ../Makefile.param

CFLAGS += -I$(PWD)/sample

SRCS := $(wildcard *.c)
SRCS += $(wildcard ./sample/*.c)
SRCS += $(wildcard $(PWD)/../common/*.c)
TARGET := sample_ive_main

# compile linux or HuaweiLite
include $(PWD)/../../../$(OSTYPE).mak
