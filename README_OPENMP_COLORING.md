# OpenMP threading via element coloring (feature/openmp-coloring)

## Summary

This branch parallelizes `compute_Fint` across CPU threads using the element
coloring from `feature/batched-force-apply`, without the AVX2 batching kernel
from that branch. Each color's elements are split across OpenMP threads,
each thread calling the original scalar `MAT_Fint` per element, so there is
no new numerical kernel here, only a change in loop order and threading.
Correctness holds at every thread count (round off level differences from
the serial baseline, same magnitude regardless of how many threads are
used). The speedup is real but modest: about 1.67x at best, reached at 4
threads, and it does not improve past that on this machine's 10 cores.

## What's here

- `SRC/color_elem.f90` -- ported from `feature/batched-force-apply` and
  trimmed. The AVX2 branch grouped each color into batches of 4 elements
  plus a remainder (`batches`, `rem`), sized off `batch_const.f90`'s
  `VEC_W`. None of that is needed for thread level parallelism, so this
  branch keeps only the flat `elem(:)` list per color and drops the
  `batch_const` dependency entirely. Same greedy coloring algorithm, same
  runtime validation (aborts if any color has a node collision).
- `SRC/spec_grid.f90` -- same 3 line hook as the AVX2 branch: `use
  color_elem`, a `coloring` field on `sem_grid_type`, and a call to
  `COLOR_build_and_validate` right after the grid is built.
- `SRC/solver.f90` -- `compute_Fint` gains an `OPT_OMP` build path. It loops
  over colors serially, and for each color opens a `!$OMP PARALLEL DO` over
  that color's element list, calling the same `FIELD_get_elem_sub` /
  `MAT_Fint` / `FIELD_add_elem` sequence the serial path uses per element.
  No batched kernel, no new material routine.
- `SRC/Makefile` / `SRC/Makefile.depend` -- compile rule now always runs
  `-cpp` (previously conditional through `OPT`, which meant the unflagged
  build wouldn't preprocess `#ifdef` at all), and dependency lines added for
  `color_elem.o`.

## Why this should be race free

Two elements in the same color never share a global node, so the scatter
step (`FIELD_add_elem`, a plain `Fout(k,:) = Fout(k,:) + fin(i,j,:)`, no
atomics) never has two threads touching the same array slot within a color.
Across colors, `!$OMP END PARALLEL DO` is an implicit barrier, so color N's
writes to `f` are all done before color N+1 starts, even though a node can
legitimately belong to elements in different colors.

`matwrk(e)` (the per element material working state that `MAT_Fint`
mutates) is a distinct array entry per element, so concurrent calls for
different elements touch disjoint memory. I checked every `save` attribute
module variable touched from `MAT_Fint`'s call chain across
`mat_elastic.f90`, `mat_plastic.f90`, `mat_kelvin_voigt.f90`,
`mat_visco.f90`, `mat_damage.f90`, `mat_mass.f90`, and `mat_gen.f90`. All of
them (`isElastic`, `isPlastic`, the `*_memwrk` memory accounting counters,
etc.) are written only once, during per-element initialization before the
solve loop starts, and are read only from inside `compute_Fint`. So there's
no shared mutable state outside `matwrk(e)` itself.

## Coloring, this mesh

Same mesh as the AVX2 branch (2.5D_inplane, 12,800 elements), same coloring
algorithm, so the result is identical: 6 colors, sizes 3200 / 3200 / 3200 /
3123 / 75 / 2. The last two colors are small. That is relevant to the
scaling result below.

## Correctness

Compared `Ux_sem2d.dat` and `Uz_sem2d.dat` (single precision SEP binary,
1610 values each) against the unflagged baseline at every thread count
(1, 2, 4, 8, 10):

| Threads | Ux max abs diff | Ux values differing | Uz max abs diff | Uz values differing |
|---|---|---|---|---|
| 1  | 1.14e-13 | 8/1610 | 1.39e-17 | 4/1610 |
| 2  | 1.14e-13 | 8/1610 | 1.39e-17 | 4/1610 |
| 4  | 1.14e-13 | 8/1610 | 1.39e-17 | 4/1610 |
| 8  | 1.14e-13 | 8/1610 | 1.39e-17 | 4/1610 |
| 10 | 1.14e-13 | 8/1610 | 1.39e-17 | 4/1610 |

Round off level, and identical across every thread count. That last part
makes sense once you think about it: within a color, no two elements share
a node, so there's no reduction or accumulation order that depends on how
many threads split up that color's work. The only reordering relative to
the serial baseline is which color an element's contribution lands in,
which is fixed by the coloring itself, not by the thread count. So the
diff pattern is set entirely by the coloring, and threading on top of it
doesn't add any further nondeterminism. No thread count showed large or
systematic differences, so I'm treating this as correct, not a bug.

## Thread scaling (2.5D_inplane, Apple M4, 4P + 6E cores, median of 3 runs)

| Build | Threads | Wall time (median) | Speedup vs OMP 1 thread |
|---|---|---|---|
| Unflagged (baseline) | n/a | 14.084 s | -- |
| Flagged (OPT_OMP) | 1  | 14.417 s | 1.00x |
| Flagged (OPT_OMP) | 2  | 9.954 s  | 1.45x |
| Flagged (OPT_OMP) | 4  | 8.642 s  | 1.67x |
| Flagged (OPT_OMP) | 8  | 8.859 s  | 1.63x |
| Flagged (OPT_OMP) | 10 | 8.891 s  | 1.62x |

OMP build at 1 thread vs the unflagged baseline: 14.417 s vs 14.084 s, about
2.3% slower. Small, and about the same size as the run to run spread in
either build (individual runs ranged 14.03-14.47 s unflagged, 14.38-15.32 s
flagged), so this looks like negligible single thread overhead from the
`OPT_OMP` code path rather than a real cost, but I wouldn't call it exactly
zero either.

## Honest read on the scaling

Speedup peaks at 4 threads (1.67x) and does not improve from there. 8 and
10 threads land at 1.62-1.63x, which is flat with 4 threads within noise,
not better. This machine has 4 performance cores and 6 efficiency cores,
so 4 threads is exactly where the performance cores run out. That's a
plausible explanation on its own, and consistent with what you'd expect
if the E-cores just don't add much for this kind of work.

But I don't think core type is the whole story, and I want to flag the
other candidate rather than just pick the explanation that sounds cleanest.
This mesh's coloring is 3200/3200/3200/3123/75/2, meaning two of six colors
have only 75 and 2 elements. Every color opens and closes an `!$OMP
PARALLEL DO` region (thread team fork, work split, implicit barrier at the
end), once per color per call to `compute_Fint`, and `compute_Fint` runs
once per timestep. At roughly 3200 timesteps for this run, that's about
19,000 parallel region launches over the full solve, a meaningful fraction
of them spawning a full thread team to process 2 elements. I have not
profiled this specifically (would need something like `perf` or Instruments
around the parallel region entry/exit to separate fork-join overhead from
actual compute time), so I can't say how much of the plateau is P/E core
saturation versus this. Both are real structural properties of this
implementation and this mesh, not measurement noise, and either one would
produce the shape of curve seen here.

Also worth being honest about scope: `compute_Fint` is the hottest part of
the solver by the earlier profiling, but it isn't the whole solver. Time
integration bookkeeping, boundary conditions, source injection, and I/O are
still serial in this build. Even with perfect scaling inside
`compute_Fint`, Amdahl's law caps the overall speedup at whatever fraction
of total wall time `compute_Fint` actually is, and I haven't isolated that
fraction on this build, so the 1.67x ceiling could be explained partly by
that too, on top of the two points above.

Net: this works, it's correct, and it's a real speedup, but it's a modest
one that saturates early, not a "throw cores at it" win. If this is worth
pushing further, I'd want to profile the parallel region overhead directly
before assuming it's core type and stopping there.

## Reproducing

    cd SRC
    make clean && make F90=gfortran OPT="-O3 -march=native -ffp-contract=fast -w" EXEC=../bin/sem2dsolve
    make clean && make F90=gfortran OPT="-O3 -march=native -ffp-contract=fast -w -cpp -DOPT_OMP -fopenmp" EXEC=../bin/sem2dsolve_omp
    cd ../EXAMPLES/2.5D_inplane
    ../../bin/sem2dsolve
    OMP_NUM_THREADS=4 ../../bin/sem2dsolve_omp

This machine is Apple Silicon (arm64), so `-march=native` targets NEON, not
AVX2. That doesn't matter here since none of this branch's code is SIMD
specific, only the thread level parallelism, which is portable.
