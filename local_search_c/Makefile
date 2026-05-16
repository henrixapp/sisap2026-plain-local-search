SHELL = /usr/bin/env bash

CC = h5cc

CFLAGS ?= -O3 -Wall -g
override CFLAGS += -std=gnu17 -march=native -fopenmp -lm -I include
override LDFLAGS += -L bin/

OBJ_SHARED := matrix.o knn.o tree.o

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