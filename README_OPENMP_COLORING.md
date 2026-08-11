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

I reran the whole sweep a second time to make sure the plateau wasn't a
fluke of one run. It reproduces: second pass gave 1.71x at 4 threads,
1.66x at 8, 1.62x at 10, same shape, all within a couple percent of the
first pass. Correctness diffs were bit for bit identical to the first
run at every thread count.

## Honest read on the scaling

Speedup peaks around 4 threads (1.67-1.71x across the two runs) and does
not improve from there. 8 and 10 threads land at 1.6-1.7x, flat with 4
threads within noise, not better.

I had two candidate explanations for this and didn't want to just pick
the one that sounded cleanest, so I tested one of them instead of
guessing. This mesh's coloring is 3200/3200/3200/3123/75/2, two of six
colors have only 75 and 2 elements, and every color opens and closes its
own `!$OMP PARALLEL DO` region once per call to `compute_Fint` (about
19,000 parallel region launches over the full solve at ~3200 timesteps).
I built a diagnostic variant with `IF(pb%grid%coloring%colors(icol)%nelem
> 64)` added to the parallel directive, so the two tiny colors just run
serially on the calling thread instead of spawning a team, and reran the
4/8/10 thread cases. Result: no measurable difference. 8 threads came in
at 8.76 s versus 8.86 s / 8.73 s for the two non-threshold runs, 10
threads at 8.96 s versus 8.89 s / 8.97 s, all inside the same run to run
noise band. Correctness was unaffected (same round off diffs).

So the tiny-color fork-join overhead is not the dominant effect (I'm not
ruling it out as a small contributor, but it isn't moving the needle at
this mesh size). That leaves the 4 performance core count as the more
likely explanation for where the plateau sits, by elimination rather than
by assumption. I did not keep the threshold change, since it tested
negative for a real effect and would just be an unexplained magic number
sitting in the code.

Also worth being honest about scope: `compute_Fint` is the hottest part of
the solver by the earlier profiling, but it isn't the whole solver. Time
integration bookkeeping, boundary conditions, source injection, and I/O are
still serial in this build. Even with perfect scaling inside
`compute_Fint`, Amdahl's law caps the overall speedup at whatever fraction
of total wall time `compute_Fint` actually is, and I haven't isolated that
fraction on this build, so part of the 1.6-1.7x ceiling is plausibly that,
on top of the P-core count.

Net: this works, it's correct, and it's a real speedup, but it's a modest
one that saturates at 4 threads, not a "throw cores at it" win, and it
looks like a genuine core-count ceiling rather than an artifact of this
particular coloring that a smarter implementation would dodge.

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
