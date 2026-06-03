# SEM2DPACK Profiling Baseline

This branch isolates the performance profiling baseline for `SEM2DPACK`.

### Environment & Methodology
- **Compiler**: `gfortran` (version 13.3)
- **Flags**: `-O3 -g -fno-omit-frame-pointer -ffree-line-length-none -pg`
- **Target OS**: Linux x86-64 (HPC / Docker environment)
- **Test case**: `EXAMPLES/2.5D_inplane` (12,800 elements, 3,208 steps)
- **Tools**: Built-in `gprof` to capture flat profiles and call graphs.

### Key Findings
Our profiling validates that **`mxmlib::mxm` accounts for only ~0.09% of the runtime**, meaning optimizing the library matrix multiplication alone yields no measurable speedup (Amdahl's Law). 

Instead, the true computational bottleneck is `mat_elast_f` (~37-44% of runtime). Under the `OPT_NGLL=5` configuration, the code dispatches to `ELAST_KD2_PSV` which bypasses `mxmlib` entirely. It relies on `My_MATMUL`, a naive 5x5 triple loop with by-value array returns that creates significant memory overhead. The secondary bottleneck is the gather/scatter operations (`field_get_elem_sub` and `field_add_elem_2`) which consume an additional ~24%.

### Reproducing the Profile
You can easily reproduce the `gprof` output by running the included script from the repository root:

```bash
# Ensure gfortran, make, and binutils are installed
chmod +x profile.sh
./profile.sh
```

Alternatively, here are the exact commands the script executes:

```bash
# 1. Setup local bin directory
mkdir -p ~/bin
sed -i 's|EXEC = .*|EXEC = ~/bin/sem2dsolve|' SRC/Makefile

# 2. Configure compiler and flags
sed -i 's|^OPT = .*|OPT = -O3 -g -fno-omit-frame-pointer -ffree-line-length-none -pg|' SRC/Makefile
sed -i 's|^F90 = .*|F90 = gfortran|' SRC/Makefile

# 3. Compile the solver
cd SRC
make clean && make
cd ..

# 4. Run the benchmark
cd EXAMPLES/2.5D_inplane
time ~/bin/sem2dsolve

# 5. Generate the flat profile
gprof ~/bin/sem2dsolve gmon.out > gprof_flat.txt
```
