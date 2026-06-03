# SEM2DPACK Profiling Baseline

This branch isolates the performance profiling baseline for `SEM2DPACK`.

### Environment & Methodology
- **Compiler**: `gfortran` (version 13.3)
- **Flags**: `-O3 -g -fno-omit-frame-pointer -ffree-line-length-none -pg`
- **Target OS**: Linux x86-64 (HPC / Ubuntu 24.04 Docker environment)
- **Test case**: `EXAMPLES/2.5D_inplane` (12,800 elements, 3,208 steps)
- **Tools**: Built-in `gprof` to capture flat profiles and call graphs.

### Key Findings (Stability Test)
To ensure robustness against scheduler noise, the solver was pinned to a single core (`taskset -c 0`) and executed 3 times back-to-back.

| Routine              | Run 1 | Run 2 | Run 3 | **Median** |
|----------------------|-------|-------|-------|--------|
| `mat_elast_f`        | 32.05%| 34.91%| 35.10%| **34.91%** |
| `field_get_elem_sub` | 24.59%| 17.41%| 22.36%| **22.36%** |
| `solve_leapfrog`     | 16.65%| 17.92%| 17.39%| **17.39%** |
| `field_add_elem_2`   | 12.02%| 13.10%| 10.34%| **12.02%** |
| `mat_elast_add_25d_f`| 6.28% | 6.34% | 5.85% | **6.28%**  |
| `mxmlib::mxm`        | 0.08% | 0.25% | 0.16% | **0.16%**  |
*Wall-clock per run: 17.47s / 16.70s / 17.57s.*

Our profiling validates that **`mxmlib::mxm` accounts for only 0.08-0.25% of the runtime** (noise floor, always last). Optimizing the library matrix multiplication alone yields no measurable speedup (Amdahl's Law). 

Instead, the true computational bottleneck is the force apply `mat_elast_f` (**~35% median**, stable #1). Under the `OPT_NGLL=5` configuration, the code dispatches to `ELAST_KD2_PSV` which bypasses `mxmlib` entirely. It relies on `My_MATMUL`, a naive 5x5 triple loop with by-value array returns that creates significant memory overhead. The secondary bottleneck is the gather/scatter operations (`field_get_elem_sub` and `field_add_elem_2`) which consume a combined **~34% median**. The gather/scatter routines are memory-bound, which explains where the majority of the run-to-run variance lands.

*(See the committed `EXAMPLES/2.5D_inplane/gprof_run[1-3].txt` files for the full verifiable raw outputs).*

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
