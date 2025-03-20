# Makefile for the diffusion simulation project

CXX := nvc++

CXXFLAGS := $(INCLUDE) -O3 -fast  -Wall -march=native -g -std=c++17

ACC := -acc -gpu=cuda12.6 -Minfo=acc


.PHONY: clean all

all: sequential parallel

sequential: seq.cpp
	$(CXX) $(CXXFLAGS) seq.cpp -o seq

parallel: par.cpp
	$(CXX) $(CXXFLAGS) $(ACC) par.cpp -o par

clean:
	rm -f seq par