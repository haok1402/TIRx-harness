# M3: end-to-end evolution run on H200 (rmsnorm_h200), 2026-11-21

**Setup**

| Item | Value |
|---|---|
| Harness | `haok/h200-harness-stack` `606b741`, checkout physically in shared workspace (`m3-stack`) |
| Kernels | dev overlay of the tvm-tirx and kernels stacks |
| Server | kcoral behind the C4 bearer-token proxy (`scripts/slurm_kcoral.sh`), one H200 |
| Agent | `claude -p`, `claude-opus-5-5`, inside a CPU-only Slurm step (`--gres=gpu:0`, 16 CPUs, `CUDA_VISIBLE_DEVICES=""`); GPU work only through kcoral |
| Limits | 20 benchmark runs (the harness sets no default budget), 2 h wall clock |

**Timeline**

| Phase | Time | Benchmark runs | Turns | Cost |
|---|---|---|---|---|
| Leg 1 | 14:46:54–15:07:54 | 4 | 74 | $4.20 |
| Leg 2 | 15:09:41–15:41:33 | 8 | 51 | $9.20 |

- **Leg 1** ended on its own at 21 min, before the planned kill. The agent believed the large shapes were at
  the HBM ceiling.
- **Leg 2** resumed the same session (`--resume`) with the remaining budget. The agent stopped with 8 runs and
  about 1 h left, judging each remaining idea worth under 1%.
- **Forced kill:** at 15:12:42 the kill script cancelled the kcoral step while a request was in flight.
  - The wrapper relaunched the server at 15:12:53 on a new port.
  - The client retried: "unreachable", then "no server URL published", then HTTP 502 twice, then
    "succeeded on attempt 5".
  - The agent continued unaware, reporting only "server preemption: happened once, recovered after ~1.5 min".

**Result**

- Best candidate `persistent-tma-ring`: **1.053x FlashInfer** (summed time over 15 shapes), 1.041x for
  batch ≥ 4113, against the **0.854x** seed. Reproduced in 3 official runs (1.051 / 1.051 / 1.053).
- Second frontier member: `register-onepass`, 1.036x.
- Main levers:
  1. Plain `ld.global.nc.v4` instead of `L1::no_allocate`, +2–3%.
  2. A sequential row stream via a global atomic ticket counter, since static persistent schedules drift.
  3. Ring retuning.
- At 16384×8192 the kernel runs at about 4.15 TB/s, against about 127 µs for a driver memcpy of the same
  bytes, so under 2% headroom is left.

**What the agent tripped on:** see `../../issues/HARNESS-001-m3-agent-findings.md`. Items 1–5 are fixed:
- token handling in the tool CLIs;
- the health check;
- `python` on the server PATH;
- the seed's `allowed_func_calls`;
- the banned-path `$PATH` false positive.

The re-run NCU check is in `ncu_check.md`.

Files: `agent_reports.md` (both final reports), `frontier_index.json`, the scripts, and the wrapper / kill
logs.
