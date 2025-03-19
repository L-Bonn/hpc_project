# Makefile for the diffusion simulation project

CXX := nvc++

CXXFLAGS := $(INCLUDE) -O3 -fast  -Wall -march=native -g -std=c++17

ACC := -acc -gpu=cuda12.6 -Minfo=acc

# Target executable name
TARGET = seq

# Source file
SRC = seq.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: clean all

all: seq par

sw_sequential: seq.cpp
	$(CXX) $(CXXFLAGS) seq.cpp -o seq

sw_parallel: par.cpp
	$(CXX) $(CXXFLAGS) $(ACC) par.cpp -o par