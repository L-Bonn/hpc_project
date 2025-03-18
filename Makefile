# Makefile for the diffusion simulation project

# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -std=c++11

# Target executable name
TARGET = seq

# Source file
SRC = seq.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean
