GPU allocation: this session runs inside a job step that owns its GPU(s)
   (Slurm sets `CUDA_VISIBLE_DEVICES={{visible_devices}}` for the step). Other
   agents' GPUs are not visible to you and must not be used.

   - Do NOT set, change, or unset `CUDA_VISIBLE_DEVICES`, and do not pick a
     GPU index yourself: run every bench / ncu / correctness-GPU-run as-is,

         python evolution/benchmark/adapter.py <workload_dir> vN
         ncu -k ... python evolution/benchmark/adapter.py <workload_dir> vN

     and the benchmark selects among the step's GPUs.
   - Do not start your own `srun`, `salloc`, or `sbatch`, and do not ssh to
     other nodes.

   Bench termination: explicit PID/PGID ownership or `timeout` only — never
   fuzzy process-name matching.
