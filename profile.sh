#!/bin/bash
# Reproducibility script for SEM2DPACK profiling (Linux/x86-64 HPC environment)
set -e

echo "Building SEM2DPACK with profiling enabled (-pg)..."

# Ensure bin directory exists
mkdir -p ~/bin

# Modify Makefile to use gfortran and enable gprof profiling
sed -i 's|^F90 = .*|F90 = gfortran|' SRC/Makefile
sed -i 's|^OPT = .*|OPT = -O3 -g -fno-omit-frame-pointer -ffree-line-length-none -pg|' SRC/Makefile


# Build the executable
cd SRC
make clean
make
cd ..

echo "Running EXAMPLES/2.5D_inplane benchmark..."
cd EXAMPLES/2.5D_inplane
time ~/bin/sem2dsolve

echo "Generating flat profile with gprof..."
gprof ~/bin/sem2dsolve gmon.out > gprof_flat.txt

echo "Profiling complete. Results saved to EXAMPLES/2.5D_inplane/gprof_flat.txt"
