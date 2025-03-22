# Makefile for the diffusion simulation project

CXX := nvc++

CXXFLAGS := $(INCLUDE) -O3 -fast  -Wall -march=native -g -std=c++17

ACC := -acc -gpu=cuda12.6,mem:managed 

MINFO := #-Minfo=acc


.PHONY: clean all

all: seq par par_v2 par_v1

seq: seq.cpp
	$(CXX) $(CXXFLAGS) seq.cpp -o seq

par: par.cpp
	$(CXX) $(CXXFLAGS) $(ACC) $(MINFO) par.cpp -o par

par_v2: par_v2.cpp
	$(CXX) $(CXXFLAGS) $(ACC) $(MINFO) par_v2.cpp -o par_v2

par_troels: par_v1.cpp
	$(CXX) $(CXXFLAGS) $(ACC) $(MINFO) par_v1.cpp -o par_v1

clean:
	rm -f seq par par_v2 par_v1