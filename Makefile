SHELL = /usr/bin/env bash

CC = h5cc

CFLAGS ?= -O3 -Wall 
override CFLAGS += -std=gnu17 -march=x86-64-v3 -fopenmp -lm -I include
override LDFLAGS += -L bin/

OBJ_SHARED := matrix.o knn.o

OBJ_KNN := main.o $(OBJ_SHARED)
OBJ_KNN := $(addprefix bin/, $(OBJ_KNN))

DEP := $(OBJ_KNN)
DEP := $(sort $(DEP))

vpath %.c src
vpath %.h include

all : KNN

-include $(DEP:.o=.d)

KNN : $(OBJ_KNN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bin/%.o : %.c
	$(CC) $(CFLAGS) -MMD -c $< -o $@

.PHONY : clean
clean :
	rm -f KNN $(DEP) $(DEP:.o=.d)