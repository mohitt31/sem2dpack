import os
import subprocess
import time
import statistics
import re
import numpy as np

def run_simulation(exec_path, workdir, use_taskset=True):
    cmd = []
    if use_taskset:
        cmd += ["taskset", "-c", "0"]
    cmd.append(exec_path)
    
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t1 = time.perf_counter()
    wall_time = t1 - t0
    
    if res.returncode != 0:
        print(f"Error running {exec_path}:")
        print(res.stdout[-1000:])
        raise RuntimeError(f"Simulation failed with returncode {res.returncode}")
    
    # Parse solver CPU time
    cpu_match = re.search(r"Total solver CPU time\s*=\s*([0-9.]+)", res.stdout)
    step_match = re.search(r"CPU time per time step\s*=\s*([0-9.]+)", res.stdout)
    
    cpu_time = float(cpu_match.group(1)) if cpu_match else wall_time
    cpu_step = float(step_match.group(1)) if step_match else 0.0
    
    return wall_time, cpu_time, cpu_step

def compare_files(f1, f2, name):
    d1 = np.loadtxt(f1)
    d2 = np.loadtxt(f2)
    
    abs_diff = np.abs(d1 - d2)
    max_abs = np.max(abs_diff)
    
    # relative diff where denom not 0
    denom = np.abs(d1)
    rel_diff = np.zeros_like(abs_diff)
    mask = denom > 1e-15
    rel_diff[mask] = abs_diff[mask] / denom[mask]
    max_rel = np.max(rel_diff)
    
    max_abs_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    max_rel_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
    
    diff_count = np.sum(d1 != d2)
    total_count = d1.size
    
    print(f"\n--- Seismogram Diff for {name} ({os.path.basename(f1)} vs {os.path.basename(f2)}) ---")
    print(f"  Total elements: {total_count}, Differing elements: {diff_count} ({diff_count/total_count*100:.2f}%)")
    print(f"  Max Absolute Diff: {max_abs:.6e}")
    print(f"  At max abs diff index {max_abs_idx}: Unflagged = {d1[max_abs_idx]:.12e}, Flagged = {d2[max_abs_idx]:.12e}")
    print(f"  Max Relative Diff: {max_rel:.6e}")
    print(f"  At max rel diff index {max_rel_idx}: Unflagged = {d1[max_rel_idx]:.12e}, Flagged = {d2[max_rel_idx]:.12e}")

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    workdir = os.path.join(root, "EXAMPLES", "2.5D_inplane")
    unflagged_exec = os.path.join(root, "bin", "sem2dsolve")
    flagged_exec = os.path.join(root, "bin", "sem2dsolve_batch")
    
    # Check if taskset is available
    use_taskset = (subprocess.run(["which", "taskset"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0)
    print(f"Using taskset -c 0: {use_taskset}")
    
    # 1. Unflagged runs
    print("\n=== Running Unflagged Benchmark (3 runs) ===")
    unflagged_walls = []
    unflagged_cpus = []
    for r in range(1, 4):
        w, c, s = run_simulation(unflagged_exec, workdir, use_taskset)
        unflagged_walls.append(w)
        unflagged_cpus.append(c)
        print(f"  Run {r}: Wall = {w:.3f}s, Total Solver CPU = {c:.3f}s, CPU/step = {s:.6f}s")
    
    # Copy seismograms
    os.system(f"cp {os.path.join(workdir, 'Ux_sem2d.dat')} {os.path.join(workdir, 'Ux_unflagged.dat')}")
    os.system(f"cp {os.path.join(workdir, 'Uz_sem2d.dat')} {os.path.join(workdir, 'Uz_unflagged.dat')}")
    
    # 2. Flagged runs
    print("\n=== Running Flagged Benchmark (3 runs) ===")
    flagged_walls = []
    flagged_cpus = []
    for r in range(1, 4):
        w, c, s = run_simulation(flagged_exec, workdir, use_taskset)
        flagged_walls.append(w)
        flagged_cpus.append(c)
        print(f"  Run {r}: Wall = {w:.3f}s, Total Solver CPU = {c:.3f}s, CPU/step = {s:.6f}s")
        
    os.system(f"cp {os.path.join(workdir, 'Ux_sem2d.dat')} {os.path.join(workdir, 'Ux_flagged.dat')}")
    os.system(f"cp {os.path.join(workdir, 'Uz_sem2d.dat')} {os.path.join(workdir, 'Uz_flagged.dat')}")
    
    med_unf_wall = statistics.median(unflagged_walls)
    med_flg_wall = statistics.median(flagged_walls)
    med_unf_cpu = statistics.median(unflagged_cpus)
    med_flg_cpu = statistics.median(flagged_cpus)
    
    print("\n=== Summary Timings ===")
    print(f"Unflagged Median Wall Time: {med_unf_wall:.3f}s (Runs: {unflagged_walls})")
    print(f"Flagged   Median Wall Time: {med_flg_wall:.3f}s (Runs: {flagged_walls})")
    print(f"Speedup (Wall-clock): {med_unf_wall / med_flg_wall:.2f}x")
    print(f"Unflagged Median Solver CPU: {med_unf_cpu:.3f}s (Runs: {unflagged_cpus})")
    print(f"Flagged   Median Solver CPU: {med_flg_cpu:.3f}s (Runs: {flagged_cpus})")
    print(f"Speedup (Solver CPU): {med_unf_cpu / med_flg_cpu:.2f}x")
    
    compare_files(os.path.join(workdir, 'Ux_unflagged.dat'), os.path.join(workdir, 'Ux_flagged.dat'), "Ux_sem2d.dat")
    compare_files(os.path.join(workdir, 'Uz_unflagged.dat'), os.path.join(workdir, 'Uz_flagged.dat'), "Uz_sem2d.dat")

if __name__ == "__main__":
    main()
