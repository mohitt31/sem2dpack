# Batched force apply (feature/batched-force-apply)

## Summary

This branch implements element coloring and a 4-wide batched Fortran version of
`ELAST_KD2_PSV`, targeting AVX2 auto-vectorization across elements. Verified correct
(matches the scalar path to round-off), but not a net win end to end at ngll=5 on
2.5D_inplane. Kept here with the diagnostics that explain why, since the coloring
infrastructure and the profiling method are reusable even though this particular
kernel change is not being merged as-is.

## What's here

- `SRC/color_elem.f90` -- greedy graph coloring on `grid%ibool`. Elements conflict if
  they share any global node. Colors are split into batches of 4 plus a remainder.
  Validated at runtime (aborts if any color has a node collision).
- `SRC/mat_elastic.f90` -- `MAT_ELAST_KD2_batched` and `MAT_ELAST_add_25D_f_batched`,
  processing 4 elements per call with a batch dimension sized to compile-time
  constants (`OPT_NGLL`, `VEC_W=4`) so all working arrays are stack-allocated.
- `SRC/solver.f90` -- `compute_Fint` gains an `OPT_BATCH` build path that loops over
  colors, gathers batches of 4, calls the batched kernel, and scatters. Falls back to
  the original per-element path for remainder elements and for any batch that is not
  homogeneous elastic-isotropic-PSV at `ngll == OPT_NGLL`.
- `EXAMPLES/2.5D_inplane/` -- benchmark and correctness diffs from the runs below.

## Results (2.5D_inplane, x86-64, AVX2, pinned core, median of 3 runs)

| Build | Median wall time | vs baseline |
|---|---|---|
| Unflagged (baseline) | 34.26 s | 1.00x |
| Color-reordered, scalar kernel | 35.21 s | 0.97x |
| Batched (OPT_BATCH) | 36.94 s | 0.93x |

Seismograms (`Ux_sem2d.dat`, `Uz_sem2d.dat`) match the baseline to round-off in every
configuration (max abs diff ~1e-13 to 1e-17; the handful of larger relative diffs are
all at near-zero values, consistent with float32 output quantization, not a logic
error).

## Why it doesn't win

Isolating the two changes (color-order vs batching) shows the coloring/reordering
itself is free (0.97x, inside run-to-run noise). The full cost is in the batched
kernel path specifically (0.93x vs the reordered-scalar 0.97x).

At `ngll=5` each element's kernel is a 5x5 contraction, small enough that gathering
4 elements' coefficients into contiguous batch buffers and scattering the result back
costs about as much as the arithmetic saved by vectorizing it. The scalar baseline is
also already efficient here: it compiles against `OPT_NGLL` as a compile-time constant
rather than a runtime size, so gfortran already optimizes and can inline it well at
`-O3`. Confirmed with `objdump` that the batched kernel does emit genuine packed AVX2
FMA instructions (`vfmaddXXXpd` across the `ymm` registers), so this is a real
gather/scatter-vs-arithmetic tradeoff, not a failure to vectorize.

## Next direction

In the original profiling pass on this branch's parent work, field gather/scatter
(`FIELD_get_elem_sub` / `FIELD_add_elem`) was close to a third of runtime, comparable
to the force apply itself. That cost looks like memory layout and indirection rather
than arithmetic, so it likely doesn't run into the same small-kernel overhead that
limited the batched force apply here. That's the more promising next target.

## Reproducing

cd SRC
make clean && make F90=gfortran OPT="-O3 -march=native -ffp-contract=fast -w"
make clean && make F90=gfortran OPT="-O3 -march=native -ffp-contract=fast -fno-vect-cost-model -w -cpp -DOPT_BATCH" EXEC=../bin/sem2dsolve_batch
cd ../EXAMPLES/2.5D_inplane
taskset -c 0 ../../bin/sem2dsolve
taskset -c 0 ../../bin/sem2dsolve_batch
