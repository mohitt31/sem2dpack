# SEM2DPACK Profiling Scripts

This branch contains the scripts used to reproduce the `gprof` profiling baseline for SEM2DPACK.

- **`profile.sh`**: Run this script on any Linux x86-64 machine (like a cluster node) to compile with `gfortran` and `-pg` flags, execute the `2.5D_inplane` test case, and generate the `gprof_flat.txt` profile data automatically. 
- Ensure you have `gfortran` and `make` installed before running the script.
