#!/bin/bash
set -e
g++ -O3 -march=native -fno-math-errno -fno-trapping-math -std=c++17 microbench.cpp -o microbench
echo "Build successful. Running benchmark pinned to core 0..."
./microbench
