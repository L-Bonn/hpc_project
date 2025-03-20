# Makefile for the diffusion simulation project

CXX := nvc++

CXXFLAGS := $(INCLUDE) -O3 -fast  -Wall -march=native -g -std=c++17

ACC := -acc -gpu=cuda12.6 

ACCtroels := -acc -gpu=cuda12.6,mem:managed 

MINFO := #-Minfo=acc

.PHONY: clean all

all: sequential parallel troels

sequential: seq.cpp
	$(CXX) $(CXXFLAGS) seq.cpp -o seq

parallel: par.cpp
	$(CXX) $(CXXFLAGS) $(ACC) $(MINFO) par.cpp -o par

troels: par_working_troels.cpp
	$(CXX) $(CXXFLAGS) $(ACCtroels) $(MINFO) par_working_troels.cpp -o par_troels

clean:
	rm -f seq par par_troels