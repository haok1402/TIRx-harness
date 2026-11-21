# NCU re-check: agent report (16:00-16:03), before the CUDA-PATH fix

See HARNESS-001 "NCU re-check" for the direct re-check after the fix.

I couldn't get any of the NCU numbers: the dedicated NCU tool can't run on the benchmark server, and even run another way, NCU isn't allowed to read the H200's performance counters. The health check and the `sys.executable` check worked. I didn't run the scoring benchmark, change the server or commit anything.

| Step | Result |
|---|---|
| 1. Health check | Passed, `status: ok` on the first try |
| 2. NCU capture with `kcoral_ncu.py` | Failed: `ncu is not installed in the remote server environment` |
| 2b. Fallback capture through `--cmd` | Failed: `ERR_NVGPUCTRPERM` (no permission to read the GPU counters) |
| 3. Kernel name, duration, DRAM %, occupancy, registers | No numbers, because no report was produced |
| 4. `sys.executable` | `/mnt/weka/home/hao.kang/kernel-evolve/wt/h-c1/.venv/bin/python3` |

**Commands that worked** (run from the worktree):
```bash
python evolution/remote/kcoral_remote.py --remote @/mnt/weka/home/hao.kang/kernel-evolve/m3/kcoral.url --health
python evolution/remote/kcoral_python.py --remote @/mnt/weka/home/hao.kang/kernel-evolve/m3/kcoral.url -- -c "import sys; print(sys.executable)"
```

**NCU command that failed:**
```bash
python evolution/remote/kcoral_ncu.py --remote @/mnt/weka/home/hao.kang/kernel-evolve/m3/kcoral.url \
  --send candidates/rmsnorm_h200/scratch/seed-ncu/send -o candidates/rmsnorm_h200/scratch/seed-ncu/seed_hs8192_bs16384.ncu-rep \
  -e HS=8192 -e BS=16384 --set full --profile-from-start off --launch-count 1 -- python capture.py
```

**What failed and why:**
1. **`ncu` isn't on the server's PATH.** It is installed at `/usr/local/cuda/bin/ncu` (version 2026.1.1). The tool looks up `ncu` on the server process's own PATH, which has no CUDA `bin` directory. Passing `-e PATH=...` can't fix this, because it only changes the environment of the program being profiled, not where the tool looks for `ncu`.
2. **The Compute Sanitizer tool probably has the same problem.** `compute-sanitizer` and `nvcc` are also missing from that PATH. I didn't run that tool, so this is an inference.
3. **The counter permission blocks NCU however it's started.** As a fallback I used the documented `--cmd` path, which goes against PROMPT.md's "use dedicated CLIs" rule, to call `ncu` by its full path. The candidate built and ran under NCU, but NCU stopped with `ERR_NVGPUCTRPERM`. The server's driver has `RmProfilingAdminOnly: 1`, which limits counter access to admins, and the server runs as `hao.kang`. Changing that needs an admin or a server launched differently.

**Fix needed on the server side** (I didn't touch it): start the server with `/usr/local/cuda/bin` on its PATH, and either run it on a node with `RmProfilingAdminOnly=0` or give the user profiling rights.

**Misleading or not matching the docs:**
- PROMPT.md says health lists `gpus[].gpu_id`, `workers[].status` and `queue_length`. The actual response has `gpu_count: 1` and a `load` object instead.
- Auth was fine: the `@kcoral.url` file and its token worked for every tool.
- `python` in the local shell already pointed to the worktree's `.venv`, so the bare `python` in the health-check loop works.
- `candidates/rmsnorm_h200` doesn't exist, so there was no seed candidate to profile. I wrote a small `solution.py` that wraps the existing kernel unchanged (`tirx_kernels.norm.rmsnorm`). I also wrote a `capture.py` that builds it the way the harness does and profiles one call of the TIRx kernel.
- The candidate built cleanly on the server with no ptxas performance warnings (no serialization or spill remarks printed).

The scripts and the fallback's output logs are in `candidates/rmsnorm_h200/scratch/seed-ncu/`. Once counters are enabled, the NCU command above should work as written.
