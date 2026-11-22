# SM90 (H200) support for TIRx: plan (v2)

Date: 2026-11-21. Author: agent sm90-plan. Builds on `docs/sm90-codegen.md` (phase 1, 2026-11-02).

## Status of the kernel-examples phase (PAUSED by the user 2026-11-22 16:05 UTC, agent sm90-plan)

New SM90 ports on dev stack v3. Each status below is the user's ruling or the independent reviewer's verdict.

| port | branch @ sha | status |
| --- | --- | --- |
| FA3 backward hd128 bf16: MHA, GQA/MQA, varlen | TIRx-kernels `haok/fa3-sm90-backward` @ `b3d9c03` (MHA `254b960`, GQA `177a8b9`) | MHA and GQA/MQA: **approved** (sm90-port-review; every output bitwise-gated via K- and q-head isolation). Varlen: **approved with documented perf misses** (user ruling): long-document causal packs 0.973-0.977, ptxas schedule sensitivity, ISSUE-022 (C); revisit with NCU or an ISSUE-022 codegen fix. hd64/hd256 in progress |
| FlashMLA sparse FP8 decode | TIRx-kernels `haok/flashmla-sm90-sparse-fp8-decode` @ `1f38b2a` | **approved with documented perf misses** (correctness approved by sm90-plan-review; port gate closed by user ruling, time-boxed like port 1). Not a pass: peak MODEL1 h128 0.909-0.913, m1c4 b256 0.974-0.977, root cause not pinned; open item to revisit if NCU access appears. Combine 55 vs 48 regs closed as a documented residual |
| DeepGEMM 1D1D: k-grouped contiguous + normal FP32-acc wgrad | TIRx-kernels `haok/deepgemm-sm90-k-grouped-fp8` @ `6be1f1b` (`9beaa71` runs on v3; `6be1f1b` drops the ISSUE-018 workaround and needs v4) | **approved** (sm90-port-review, with notes resolved). Residual spill overs at the setmaxnreg edge filed as ISSUE-025 (perf-neutral) |

v4 dev stack: tvm-tirx `haok/sm90-stack-fixes-v4` @ `d1080de9e` = v3 + ISSUE-016, 017, 018, 021, 024 (venv `fixstack4fenv`). ISSUE-023 parked (B2 4096x32768x512 -6%); ISSUE-022 and 025 open.

Final-v4 re-pass (generated CUDA v3 -> v4 differs in every kernel only by ISSUE-024's header block, plus ISSUE-017's
`&&`/`||` -> `&`/`|` in MQA logits, FlashMLA dense/sparse and FA3 bwd): bitwise and the official runner re-passed for
all ten kernels (MQA logits at both 2da8159 and the fence commit 49dbee6).

Follow-ups recorded, no action now:
- B2 `fp8_blockscale_gemm_sm90` config `1000x24584x1520` spills 64 B stores / 68 B loads (48 B stack) and fails the
  official runner under `perf_remarks=error` (19/20). Pre-existing: identical on v3 (fixstack3env), so not a v4
  regression.

### Resume (paused 2026-11-22; lenient pause, nothing new started)

All branches below are local; the user pushes. Dev stack for everything: tvm-tirx `haok/sm90-stack-fixes-v4` @
`d1080de9e`, venv `/mnt/weka/shrd/ci/haok/TIRx-kernels/fixstack4fenv` (fixstack4env = the pre-ISSUE-024 v4, kept).

| agent (tmux `agent-<name>`) | branch @ sha | state | next step on resume | resume note |
| --- | --- | --- | --- | --- |
| port-fa3bwd90 | TIRx-kernels `haok/fa3-sm90-backward` @ `8fcd903` (**wip** hd64 commit on `f3b648b` hd64 sketch, on final `b3d9c03`) | MHA/GQA/varlen final at `b3d9c03`; hd64 descriptor change (user-approved: cute advance form, hd64 only) is in the wip commit, hd128 PTX verified byte-identical 28/28 | re-run hd64 bitwise / runner / CUPTI on v4; the one causal-hd64 trace experiment, report before fixing; then commit hd64 properly and send to sm90-port-review; then hd256 | `wt/k-fa3bwd90/.porting/fa3_backward_sm90/RESUME.md` |
| port-mqalogits90 | TIRx-kernels `haok/deepgemm-sm90-mqa-logits-perf` @ `49dbee6` (fence commit on `2da8159`) | both pass full bitwise (378) and runner (65/65, 75/75) on final v4; fence SASS = add.u64 pin; small-decode with ISSUE-024: b2 h64 0.979 -> 0.987, b1 h32 0.990 -> 0.999 | attrs_ab on b1 n1 h64 mp512 / mp160, full report, PR text update; then `49dbee6` to a reviewer (sm90-plan-review re-time) | `sm90-probe/mqa90/RESUME.md` |
| port-dgkgroup90 | TIRx-kernels `haok/deepgemm-sm90-k-grouped-fp8` @ `6be1f1b` | approved; re-passed on final v4 | none (push). Optional re-check after ISSUE-022/025 | `wt/k-dgkg90/RESUME.md` |
| port-fmlasparse90 | TIRx-kernels `haok/flashmla-sm90-sparse-fp8-decode` @ `1f38b2a` | approved with documented perf misses; re-passed on final v4 | none. Optional: producer-loop stall attribution if NCU / PC sampling appears | `wt/k-fmla90s/RESUME.md` |
| perf-fa3fwd90 | TIRx-kernels `haok/sm90-fa3-causal-perf` @ `a38713f` (`33434b5` + comment) | B3 causal approved at `33434b5`; ISSUE-017 drop measured and kept the folded mask; re-passed on final v4 | sm90-plan-review re-times `a38713f` (comment-only delta), then address findings | `sm90-probe/fa3c/RESUME.md` |
| sm90-port-review | none (review worktrees `rv-fa3bwd` @ `b3d9c03`, `rv-dgkg` @ `6be1f1b`) | all verdicts final | review FA3 bwd hd64 when routed | `review-ports/RESUME.md` |
| sm90-plan-review | none | all verdicts final (ISSUE-024 approved at `4ec7a4aa9`). Its note says v4 @ `63de01a30` and next queue ISSUE-019/020: both stale; v4 is `d1080de9e`, 019/020 were closed without compiler branches | next compiler fix from sm90-codegen-fix (ISSUE-022) | `~/sm90-plan-review/RESUME.md` |
| sm90-codegen-fix | tvm-tirx fix branches below; ISSUE-022 work in progress (branch not reported) | was mid-turn on ISSUE-022 when paused; pause message queued, reply pending | finish / report ISSUE-022 (A row base, B descriptor rebuild incl. the FA3 hd64 evidence, C varlen schedule); ISSUE-025 info only | not yet written |
| sm90-plan (this agent) | TIRx-harness `haok/sm90-plan` (this doc) | v3 -> final-v4 source diff and re-pass done; no steps running; loop cancelled | route FA3 hd64 and MQA `49dbee6` to reviewers when they report; collect sm90-codegen-fix's pause note | this section |

tvm-tirx branches (all local): v4 `haok/sm90-stack-fixes-v4` `d1080de9e`; fixes `haok/fix-arith-fence` `0b0dfcac4`
(016), `haok/fix-bool-bitwise` `bd6eee63f` (017), `haok/fix-ld-param` `5cf844d21` (018), `haok/fix-warp-pure`
`d161ce94f` (021), `haok/fix-nvrtc-libcu-payload` `4ec7a4aa9` (024), parked `haok/fix-bind-refold` `ead6566d1` (023).
v4 also carries one local test rewrite (`63de01a30`'s any_sync test for v3's call-site binding) that is not on any fix
branch.

Scratch state: the generated-source diff is in `/mnt/weka/shrd/ci/haok/TIRx-kernels/v4diff` (v3 vs final v4), with
`v4diff.v4_pre024`, `v4diff.memo_bug` (invalid, cache bug) and `v4diff.v4a_partial` kept for reference; scripts in
`sm90-probe/v4diff/`.

## Final status of the ports phase (2026-11-22, agent sm90-plan)

**Goal (from the coordinator):** check and validate TIRx's correctness on a few more kernels, through faithful ports
of state-of-the-art SM90 kernels under TIRx-kernels `ported/<upstream>/`. Each port must meet five checks:
- correct against upstream on upstream's own shapes plus edge shapes;
- the official runner with nothing skipped;
- `perf_remarks=error` clean;
- CUPTI speed against upstream;
- any compiler issues found filed in `sm90-probe/issues/`.

**Status: done.** Three ports are complete and reviewer-approved.

### The three ports

| Port | Upstream | Branch(es), head | Correctness | Official runner (`perf_remarks=error`) | Speed vs upstream |
|---|---|---|---|---|---|
| **1. DeepGEMM SM90 FP8 1D2D m-grouped GEMM** (contiguous, psum, masked) | DeepGEMM `559d79fb` | base `haok/deepgemm-sm90-m-grouped` `251c19e`; coverage extension `haok/deepgemm-sm90-m-grouped-coverage` `c714afa`; dev-stack follow-up `haok/deepgemm-sm90-m-grouped-v2stack` `b0ce824` | 89/89 configs bitwise identical over the whole output, with a NaN sentinel. Upstream's 48 test calls plus edge and coverage shapes; the review ran 20 more of its own | 58/58 + 31/31, 0 skipped | Bench-suite matrix, v2 stack, 66 shapes: geomean 1.001, 61/66 >= 0.99. Coverage shapes 0.994 (18) and 0.996 (10, cluster-1) |
| **2. FlashMLA SM90 dense MLA decode** (agent port-flashmla90) | FlashMLA | `haok/flashmla-sm90-dense-decode` `2af43a2` | `out` and `lse` bitwise identical to `flash_mla_with_kvcache` | 345/345, 0 skipped | Its PR_DESCRIPTION.md: geomean about 0.99-1.0 over 22 configs |
| **3. DeepGEMM SM90 FP8 MQA logits + paged MQA logits** (agent port-mqalogits90) | DeepGEMM `559d79fb` | `haok/deepgemm-sm90-mqa-logits` `1daf758` | bitwise identical over the whole logits storage (normal-range data; see its PR text for the contraction note) | 65/65 + 75/75, 0 skipped | Its PR_DESCRIPTION.md: worst 0.949, geomean 0.980 (dense bf16 lowest) |

Each port went through the porting skill:
- **Sketch:** a line-info export of upstream, then a sketch, then the sketch reviewer.
- **Correctness:** an independent correctness reviewer.
- **Final check:** the external reviewer.

The details are in each branch's PR_DESCRIPTION.md.

**Port 1 in detail.**
- **Scope:** every specialization class DeepGEMM's SM90 heuristic picks for realistic MoE shapes. That is 35
  classes:
  - multicast on A or B, or none;
  - BLOCK_N 32/48/64/96/112/128/160/192;
  - BLOCK_M 64/128.
- **Reach:** all 378 shapes of the review's sweep and all 552 of `sm90-probe/dg90/reach.py`
  (DeepSeek-V3/V3.2 and Qwen3-MoE, EP 8-64). The review found 234/234 tile picks equal to DeepGEMM's
  `DG_PRINT_CONFIGS`.
- **ISA fidelity:** per class, every compute-instruction count equals upstream's line-info export, with no `.ftz`.
  Two producer-side differences are documented; neither changes results.
- **Perf work:** recorded in the ledger `.porting/sm90_fp8_gemm_1d2d/perf_ledger.md`.
  - **E3 (adopted):** the invalid-band skip loop as upstream's constant-trip unrolled loop. It took the K-loop
    from 146 to 135 SASS instructions per iteration and the geomean from 0.989 to 0.996.
  - **Rejected, with evidence:** E1, E2, E4 and E5.
- **SM-count bug, found here:** `runner.hardware_num_sms()` fell back to B200's 148 before CUDA was initialized.
  - **Port fix:** `251c19e`.
  - **Runner fix (plan B-b):** `haok/runner-hardware-num-sms` `29528ea` on TIRx-kernels, reviewer-approved. The
    runner queries the device in a child process.

### Compiler issues filed in this phase, and their state

- **ISSUE-010 (fast-math default, FTZ):** fixed, `fp_mode`.
- **ISSUE-011 (uint32-to-int32 cast assumed non-negative):** fixed.
- **ISSUE-012 (hand-written mbarrier wait blocks unrolling):** fixed, `mbarrier_wait_parity`.
- **ISSUE-013 (NaN self-compares folded):** fixed.
- **ISSUE-014 (nvcc `-arch sm_90a` adds `compute_90`):** fixed.
- **ISSUE-015 (float cancellations such as `x - x -> 0` should not fire under `fp_mode="precise"`):** filed by
  sm90-codegen-fix; open, low priority.
- **HARNESS-002:** kcoral's sealed-memfd mmap fails on Linux < 6.7, which breaks every bench-suite request over
  1 MiB. A step-local workaround is in `sm90-probe/kcoral_compat`; the fix belongs in kcoral.

**End-to-end validation of the fixes.**
- **Port 1:** on v2 it dropped its ISSUE-010/011/012 workarounds, with 79/79 bitwise and no speed loss. The two
  shapes that trailed DeepGEMM reached parity.
  - On v3, the rebased follow-up `b0ce824` gives 89/89 bitwise, including the 10 cluster-1 classes.
  - Its official runner gives 58/58 + 31/31, 0 skipped.
- **Finding for ISSUE-010:** `fp_mode="precise"` alone does not reproduce upstream's factored instruction form for
  the BLOCK_N 48/112 promotion. The port keeps that form, written in plain TIRx.
- **Ports 2 and 3:** each validated its own workaround drops on v2 and v3 (see their PR texts).

### tvm-tirx dev stacks (built in this phase; separate wheel and venv each)

| Stack | Branch @ head | Content | Venv |
|---|---|---|---|
| v1 | `haok/sm90-stack-fixes` `801280ac4` | track A items + ISSUE-001..003 | `TIRx-kernels/b3env` |
| v2 | `haok/sm90-stack-fixes-v2` `1c292fdf4` | v1 + ISSUE-010..014 + A2/A3/A4/A5 | `TIRx-kernels/fixstack2env` |
| v3 | `haok/sm90-stack-fixes-v3` `4865655a5` | v2 + ISSUE-005..009 + A3 with per-toolchain floors (NVRTC 13.2 and 13.4) | `TIRx-kernels/fixstack3env` |

**Codegen suite** (tvm-tirx `tirx/codegen`, `tirx/tile`, `backend/cuda` and `sym`, one H200, compared against the
previous stack on the same test tree):
- **No regression:** nothing fails on a newer stack that passed on an older one.
- **v3:** 3456 passed and 42 failed. All 42 are `tirx/tile/cuda` tests that load PTX through the driver. The 595.91
  driver cannot JIT PTX from NVRTC 13.4 (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`), so this is environmental.
- **A3 on v3:** active with the NVRTC 13.4 data, and it raises nothing on any port.

### Merge order (all are local or pushed topic branches; the user pushes)

1. **tvm-tirx track A**, as in the 2026-11-21 status below, with A3 now carrying `b58c1a791` (one floor data set per
   toolchain). Then sm90-codegen-fix's `haok/fix-*` for ISSUE-004..014.
2. **TIRx-kernels:**
   1. `haok/runner-hardware-num-sms` (B-b), independent of the ports;
   2. `haok/deepgemm-sm90-m-grouped`, then `haok/deepgemm-sm90-m-grouped-coverage` on top of it;
   3. `haok/flashmla-sm90-dense-decode` and `haok/deepgemm-sm90-mqa-logits`, independent of each other.
3. **Follow-ups once the dev-stack fixes are on tvm-tirx main and in the pinned wheel:**
   - `haok/deepgemm-sm90-m-grouped-v2stack`, which drops port 1's compiler workarounds;
   - the ports' local SM-count defaults, which can go once B-b lands.

### Open follow-ups

- **Port 1 speed:**
  - 5 of 66 bench-suite shapes are at 0.986-0.988 (v2), mostly small kernels. The skill's "> 0.99 on every shape"
    is not met for them.
  - **Suspect:** per-tile code outside the K loop. In the SASS, TIRx has more SYNCS, NANOSLEEP and UMOV sites than
    DeepGEMM in masked 128x192.
- **Port 1 ISA:** NVRTC does not unroll the producer K loop as nvcc does upstream. It also does not fold the plain
  A/SFA branch when multicast is on A. Forcing the unroll gained nothing.
- **Bench suite:**
  - CPU prepare is unusable on the kcoral GPU worker, whose CUDA is already initialized. Every workload falls back
    to GPU prepare.
  - HARNESS-002 needs a kcoral fix.
- **The review's 0.83x on contiguous 128x160:** an order effect of its alternating loop, unexplained. Our runs on
  the fixed branches give 0.974-0.976 (v1) and 0.986-0.988 (v2/v3), warm or cold.
  - The earlier cold runs that printed nothing were the SM-count bug. At 148 SMs that shape picks 128x144, a class
    the port refuses.
- **Driver:** NVRTC 13.4 PTX can't be JIT-loaded by the 595.91 driver. This affects only PTX-format test builds.
- **Shared space kept for reruns:** `TIRx-kernels/{refenv,b3env,fixstack2env,fixstack3env,mqa90env}`, the tvm-tirx
  wheels and build dirs, `dg_jit_cache`, and `bench_suite_runs`.

## Final status (2026-11-21, agent sm90-plan)

**Done:** M1–M4. **Track A finished (2026-11-21), awaiting review:**
- **A4** (DSMEM bulk-copy tile pattern) and **A5** (mma.sync/ldmatrix `KeyError 'm'`) are pushed. They are
  stacked on A2, whose strict xfails they flip.
- **A3** (`VerifyPTXTarget`, per-variant arch floors) is committed on `haok/ptx-arch-floor`
  (`f6bfa25d5`), off upstream main. Its validation is in that branch's PR_DESCRIPTION.md.

A9 stays deferred.

### Branches (all pushed) and their upstream targets

| Repo → upstream `main` | PR branches, in merge order | Dev-only (do not merge) |
|---|---|---|
| **tvm-tirx** | 1. `haok/sm90-wgmma-fence` (A1)<br>2. `haok/sm90-test-gate` (A2)<br>3. `haok/sm90-ptxas-log` (A10)<br>4. `haok/sm90-wgmma-helpers` (A7)<br>5. `haok/sm90-pipeline-multicast` (A6)<br>6. `haok/sm90-packed-f32` (A8)<br>7. `haok/sm90-setmaxnreg-contract` (A11, needs A10)<br>8. `haok/dsmem-pattern-fix` (A4) and `haok/mma-pattern-m` (A5), both after A2<br>9. `haok/ptx-arch-floor` (A3, off main; not pushed yet)<br>Then sm90-codegen-fix's `haok/fix-*` for ISSUE-001..007, and `haok/fix-a7-uniform-warpgroup-docs` after A7. | `haok/sm90-stack` (A-items, cherry-picked); `haok/sm90-stack-fixes` (stack + ISSUE-001..003, B3's build); stale: `haok/sm90-plan`, `haok/sm90-design` |
| **TIRx-kernels** | 1. `haok/test-runner-skips` (all-skipped exits 3)<br>2. `haok/rmsnorm-int64-offsets` (sm_100 64-bit offsets)<br>3. `haok/sm90-rmsnorm` (B0; contains 2)<br>4. `haok/sm90-gemm` (B1)<br>5. `haok/sm90-fp8-gemm` (B2)<br>6. `haok/sm90-fa3` (B3) | `haok/sm90-kernels-stack` |
| **TIRx-harness** | 1. `haok/h200-bench-stack` (C1)<br>2. `haok/h200-target-arch` (C2)<br>3. `haok/h200-tasks` (C3, stacked on C2; seeds need the TIRx-kernels B-branches)<br>Independent of each other: `haok/srun-local-mode` (C5), `haok/kcoral-slurm` (C4), `haok/dev-local-stack` (C6 overlay), `haok/sm90-skills` (C7), `haok/wgmma-fail-soft` (C8 phase 1), `haok/banned-path-list` (C9). | `haok/h200-harness-stack`; docs: `haok/sm90-plan` |

**Cross-repo dependencies.**
- **B1/B2** need tvm-tirx A1, A7, A6 and A11 (A10 for the `perf_remarks` checks).
- **B3** also needs ISSUE-003's `broadcast_warpgroup_id`, i.e. `haok/fix-cuda-warpgroup-uniform`; it
  binds the value at entry because of ISSUE-008.
- **Pins:** C6 is a dev overlay until tvm-tirx/TIRx-kernels releases contain the above. The pin bump
  (plan C6) is still to do.

### Verified on H200 (driver 595.91.07)

**Kernels** (official `tirx_kernels.test` in the pinned-reference venv: 0 skipped)

| Kernel | Tests | Gate result |
|---|---|---|
| B0 rmsnorm | 41/41, incl. > 2^31 elements | ≥ 1.010x FlashInfer per shape at bs ≥ 4113; 1.033x summed |
| B1 GEMM | 38/38 | 0.989x bf16 / 0.993x fp16 vs cuBLAS |
| B2 FP8 GEMM | 20/20 | 0.997–0.998x DeepGEMM, with the proxy fence |
| B3 FA3 forward | 20/20 | **CUPTI kernel-only:** worst gated 0.926x, geomean 0.975x vs FA3 |

All four are `perf_remarks="error"`-clean.

**Harness**
- **Tasks:** all four H200 tasks run with seeds, kernel-only CUPTI timing.
- **M3:** an evolution run (rmsnorm_h200, 1.053x) survived a forced kcoral kill.
- **kcoral tools over the token proxy:** health, python, sanitizer and IKET work. NCU is blocked by
  the driver (below).
- **NumSim / synccheck / racecheck** on the WGMMA kernels.
- **evolution/tests:** 239 pass.

### Needs a B200 (sm_100) run

- **TIRx-kernels `e8e7733` (rmsnorm sm_100 64-bit offsets):** timing; static SASS is +0–8% integer
  ops. Also its new > 2^31-element configs.
- **tvm-tirx:**
  - A1, A2, A10, A8: the full GPU suites on sm_100. A8's packed path is only compile-checked there.
- **TIRx-harness:**
  - C2: B200 specs render byte-identical by test; run one B200 task end to end.
  - C8: run tools/tests on B200 for the sm_100 corpus.
  - C1 timer changes on B200 nodes.

### Open follow-ups

- **sm90-codegen-fix:**
  - ISSUE-008: `broadcast_warpgroup_id` re-emitted per use; a divergent shfl hang.
  - ISSUE-009: float `T.max` is 3 instructions, and the exp2 add+mul; together 20% of B3's
    instructions.
  - ISSUE-001..007 fix branches still to land upstream.
- **B3:**
  - a persistent scheduler for short causal tiles, which must rework the unmatched final ping-pong
    `bar.arrive` and the epilogue's Q-buffer reuse;
  - 176-row KV blocks.
  - No evolution run on `fa3_forward_h200` yet.
- **NCU:** the nodes set `RmProfilingAdminOnly=1`. Only admins can lift it
  (`NVreg_RestrictProfilingToAdminUsers=0`); this goes to the user.
- **A3/A4/A5 review**, and the C6 pin bump.
- **Housekeeping:** HARNESS-001 #6: the wiki reference checkout lacks the SM90 kernels. The two
  stale M3 run dirs were deleted on 2026-11-21, as approved.
  - Shared-space venvs `TIRx-kernels/{refenv,b3env}` and the tvm-tirx wheel and build dir are kept
    for reruns.


**v2 changelog (2026-11-21).** v2 applies the independent review (agent sm90-plan-review,
`~/sm90-plan-review/REVIEW.md`, findings F1–F10 plus the named-barrier gap) and the coordinator's rulings. The
v1 plan is in git history.

- **A10 (new, F1).** TIRx drops the NVRTC/ptxas log. A10 surfaces C75xx WGMMA-serialization, spill and
  stack-frame lines as warnings, with an error mode. It comes right after A2 and before any kernel.
  `sass_check.sh` is fixed: it now greps `C75xx` and checks chain structure.
- **Targets (F2, F3).**
  - Every perf target is a same-step, interleaved, L2-flushed ratio.
  - Absolute TFLOPS are informational only, with node and clocks.
  - Every bench record logs the SM clock and throttle reasons.
  - The probe GEMM is restated as **0.86–0.95x on large-K squares and 0.64–0.78x on LLM shapes** (was 0.87–0.96x).
- **B1 gate (F3).** 0.95x geomean over shapes with ≥ 2 waves of tiles. Other shapes are tracked, not gated.
  Small-tile and split-K are out of scope.
- **Tile designs (F4).** B2 gets its own tile/register design item (two accumulators). B1, B2 and B3 must
  each be C75xx-clean under A10's error mode.
- **A7 tests (F5, named-barrier gap).**
  - Chain structure: one `WARPGROUP.ARRIVE` per chain, `gsb0` only on the last GMMA, no C75xx.
  - ss and rs chains of 4 k-steps.
  - A `NamedBarrier` helper (id 0 reserved) with a 2-consumer-warpgroup test.
  - A one-warpgroup FA-forward microkernel, so B3's risks surface in track A.
- **A6 (F7).** A `max_active_clusters` host helper sizes persistent grids.
- **A2(d) (F8).** Each threshold is mapped on its own; `test_ptx_dialect.py:3330` keeps running on sm_90.
- **A1 (F6).** The by-reference half is described as defensive: it matches CUTLASS, with no observed SASS effect.
- **Kernel order.** B0 is a smoke test only. The "couple of SM90 examples" are **B1 + B2**, and B2 comes
  before B3. FA3 is measured at every B3 gate shape before B3's target is fixed (F9).
- **§9 open questions settled.**
  - Helpers go in tvm-tirx `backend/cuda/lang`.
  - Development uses our pair and does not block on 0.1.1.
  - NumSim phase 2 is out of scope.
  - Authoring is tirx-lite + raw PTX + A7; A9 is deferred.
  - Asking for an upstream Hopper CI lane is the user's call. We run A2 weekly regardless.
- **F10.**
  - C2: 31 of 32 YAMLs say "B200 (sm_100)", not all 32.
  - C1: the CUDA-event fallback must flush L2 or refuse to score.
  - Tree comparisons must bypass the editable finder (appendix).
- **A11 (added after A10):** the `setmaxnreg` launch contract, found through A10's C7508 remark.
- **v2 check (reviewer, folded in):** the B1 gated set is now listed explicitly (8192x1024x2048 is 256 tiles,
  not 32); A6 acceptance compares against the raw driver query; every bench record logs the hostname.

Hardware for every number here: 1x H200 (driver 595.91.07, CUDA 13.2 toolkit on the node), 1-GPU Slurm
steps of PithTrain task 2394207_98. New evidence: `sm90-probe/logs/plan/` (index in the appendix).

## 0. Verdict: is PTX codegen the missing piece?

**No, and HEAD shows the same.** The PTX table already emits every sm_90a instruction fast kernels
need. On tvm-tirx HEAD (`6569e2a18` plus the fix below) the phase-1 probes reproduce unchanged:

- WGMMA bf16/fp16/fp8, ss and rs forms: 10/10 cases correct (`wgmma.log`).
- Warp-specialized GEMM: bit-identical to cuBLAS. With fair timing (interleaved, L2-flushed, same step;
  review F2) it reaches **0.86–0.95x on large-K squares and 0.64–0.78x on LLM shapes**. The `gemm.log`
  ratios of 0.87–0.96 came from an order-sensitive timer and are retired.
- DSMEM, PDL and clusters: 6/6 (`cluster_pdl.log`).
- tirx-lite TMA pipeline with `specialize`: correct (`txl.log`).

What actually blocks SM90 work is, in order of cost:

1. **Kernels.** Every MMA kernel in TIRx-kernels is built on tcgen05/TMEM, so each must be **re-tiled**
   for SM90, not recompiled. None of the 106 registered kernels lists `sm_90a`.
2. **Helpers above PTX.** tirx-lite has no WGMMA descriptor or chain helper. Packed f32x2 math is
   sm_100-only. SM counts are hard-coded to 148.
3. **Harness.** Tasks have no arch field. Every spec, skill and most baselines are Blackwell-only.
   NumSim/synccheck reject WGMMA. The DeepGEMM wheel does not load on these nodes.
4. **Compiler.**
   - One codegen bug: the WGMMA accumulator fence. It is fixed on our branch but not upstream.
   - ptxas WGMMA-serialization (C75xx) and spill messages are dropped, so the worst SM90 performance bug
     is silent (review F1 → A10).
   - A test gate that runs **0 of 281** Hopper/TMA tests on an H200 (`upstream_gate_asis.log`).
   - Two test files that crash at collection on any sm_90a device.
   - A missing arch check.
   - Two tile-pattern bugs that affect every arch.

So the codegen track is short, and it comes first only because the kernels depend on it.

## 1. Upstream since 2026-11-02, and re-verification

### 1.1 What changed that matters for SM90

| Repo (old → new HEAD) | Change | SM90 impact |
|---|---|---|
| tvm-tirx `3ad801cb` → `6569e2a18` (88 commits, tag **v0.1.0** = `f43b3030d`, 2026-11-12) | #232 "Pass compilation targets explicitly and remove target scopes" | `with target:` is gone. Callers pass `target=`/`backend_config`. Our `probe_txl.py` was updated. |
| | #193 "Align device target architecture detection" | `Target.from_device` now gives `sm_90a` (not `sm_90`) on H200. Good: wgmma/setmaxnreg need the `a` target. |
| | #241 PTX ISA 9.5 / sm_107 entries; table certified with CUDA 13.5 ptxas | No SM90 instruction changes. `cert_arch` is still per family (maximum floor), so the arch-check design below still applies. |
| | Native `backend/cuda/lang/{pipeline,tile_scheduler,clc,alloc_pool}.py` (pipeline helpers moved into tvm-tirx) | `MBarrier`/`TMABar`/`PipelineState`/remote arrive are sm_90-valid. Only `TCGen05Bar` and the CLC scheduler are sm_100. SM90 helper work now lands partly in **tvm-tirx**, not only tirx-lite. |
| | #184 void result types for `wait_until`/timers | Removes a silent statement-drop bug that hit tirx-lite on every arch. |
| | Not changed: test gate (`testing/_pytest.py:136`, `tile/cuda/conftest.py`), `cpp/asm.py` noop_barrier, `async_copy.py` DSMEM pattern, engine arch check | All phase-1 compiler gaps are still open upstream. |
| | Build: tvm-ffi 0.1.16 submodule; `mlc-z3-static` is now a build requirement | On gcc 11 nodes Z3 must still be OFF. Run `git submodule update` before rebuilding. |
| TIRx-kernels `716d8a3` → `16f05fc` (8 commits) | Adapted to tvm-tirx v0.1.0 (#19, #21). #24 removed `tirx_lite/pipeline.py` and now re-exports the native tvm-tirx pipeline classes. #15 added an FA4 tile-op kernel, which fails lowering on every arch. | **The rmsnorm version skew from phase 1 is gone** (`rmsnorm_trace_sm100_control` OK on HEAD). Still no sm_90a kernel. Pins `apache-tvm-tirx==0.1.0`, which lacks the WGMMA fence fix. |
| TIRx-harness `31aafff` → `7ec7e40` (21 commits) | #45 moved the stack to CUDA 13.4: torch 2.15+cu134, `nvidia-cuda-nvrtc>=13.4`, `cupti-python==13.4.0`. Pins `apache-tvm-tirx==0.1.0` and `tirx-kernels==0.1.3`. The rest is sm_107 NumSim modelling. | Nothing SM90. The CUDA 13.4 pins looked like a blocker for a 13.2 driver. **They are not** (§1.2). |

### 1.2 Phase-1 claims re-checked on HEAD

| Claim (phase 1) | On HEAD | Evidence (`sm90-probe/logs/plan/`) |
|---|---|---|
| PTX table covers sm_90a; WGMMA all dtypes ss/rs correct | **Holds** | `wgmma.log` 10/10 |
| Raw-TIRx WS GEMM at 0.87–0.98x cuBLAS | **Bit-identical, yes. The ratio was shape-selected and order-sensitive** (review F2). Fair, interleaved timing gives 0.86–0.95x on large-K squares and 0.64 (8192x8192x1024) / 0.78 (4096x7168x2048) on LLM shapes. Absolute cuBLAS numbers vary by node and power cap (8192³: 693–702 vs 786 TFLOPS; SW power-cap throttling at about 1500 MHz). | `gemm.log` (retired); review `logs/gemm_fair.log`, `interact.log`, `smi_interact.csv` |
| DSMEM / PDL / preferred cluster / host cost | **Holds**: 6/6, 4.2 us per call vs 4.9 for torch | `cluster_pdl.log` |
| noop_barrier bug; fixed on branch | **Still unfixed upstream.** Cherry-picks cleanly: `tvm-tirx@haok/sm90-plan` `5bdc1cc07`. `test_wgmma_ss_nt` passes with the fix. | `upstream_hopper.log` |
| Test gate skips all GPU tests below sm_100 | **Holds**: 281/281 skipped on H200 | `upstream_gate_asis.log` |
| — (new) `test_cuda.py` and `test_ptx_dialect.py` cannot even be collected on sm_90a | **New bug**: module-level `int(env.cuda_arch()[3:6])` gives `int('90a')`, a ValueError (`test_cuda.py:1312`, `test_ptx_dialect.py:3132,3330`) | `upstream_broad_classified.txt` |
| DSMEM bulk-copy tile pattern broken (all archs) | **Holds** (`test_dsmem` 4/4, plus `test_pattern_transfers::test_dsmem_remote_addresses`) | same |
| mma.sync tile pattern `KeyError 'm'` (all archs) | **Holds** (6 tests) | same |
| `test_wgmma_rs_nt` test bug | **Holds** | same |
| Broad suite on sm_90a: failures are hardware limits plus a few real bugs | **Holds**: 809 pass / 66 fail / 297 skip. The 66 failures: 44 hardware limits (26 interleaved/fp4 tensor maps, 18 ptxas `cta_group`/`gather4`/`f32x2`); 6 collection errors (2 are the new bug above, 4 are artifacts of copying the tests); 6 mma `'m'`; 5 DSMEM; 5 test artifacts or test bugs. 297 tests are skipped, 275 of them by explicit upstream "Blackwell required" markers (phase 1 had 52 skips). | `upstream_broad.xml`, `upstream_broad_classified.txt` |
| TIRx-kernels HEAD does not trace against tvm-tirx HEAD | **Fixed upstream.** 95/106 kernels trace and compile at their own arch; the 11 failures are environmental (§6.3) | `pair.log` |
| Harness CUDA 13.4 stack vs a 13.2 driver | **Works**, by CUDA minor-version compatibility. torch 2.15+cu134 matmul and `torch.compile`, NVRTC 13.4 compiling and launching a TIRx sm_90a kernel, and CUPTI 13.4 activity records all run on 595.91.07. | `stack.log` |
| DeepGEMM baseline wheel | **Does not load**: it needs GLIBC 2.38 and the nodes have 2.35. A source build of the pinned commit `559d79f` works after changing `-std=c++17` to `-std=c++20`, because torch 2.15 headers require C++20. | `stack.log`, `deepgemm.log` |

## 2. Track A — tvm-tirx compiler (each item is one upstream PR)

Sizes: S < ~100 lines plus tests; M = a few hundred lines in one module; L = new subsystem.
"H200 gate" = the acceptance test, run as a 1-GPU step with `pytest` on the tvm-tirx tree. After
A2 lands, no plugin is needed.

| # | Item | Size | Files | H200 acceptance test |
|---|---|---|---|---|
| **A1** ✅ | **WGMMA accumulator fence fix.** `wgmma_noop_barrier` crashed codegen (`'TensorLoad' object has no attribute 'dtype'`). The helpers now also take the register by reference. That half is **defensive**: it matches CUTLASS `warpgroup_fence_operand`, and the review saw no SASS effect (F6). **Done:** branch `haok/sm90-wgmma-fence` `e8e44225a`, pushed. | S | `backend/cuda/cpp/asm.py`, `tests/python/tirx/codegen/test_hopper.py` | `test_wgmma_ss_nt` passes; `test_wgmma_noop_barrier_fences_register_by_reference` (CPU source check) passes; `probe_wgmma.py` 10/10 (`logs/a1/`) |
| **A2** | **Let SM90 tests run on SM90.** (a) Replace the blanket "compute ≥ 10.0" skip in `codegen_collection_modifyitems` and `tile/cuda/conftest.py` with a per-test minimum: new marker `pytest.mark.cuda_min_compute(9, 0)`. Unmarked GPU tests keep requiring 10.0, so B200 behaviour is unchanged. (b) Mark the Hopper, DSMEM, TMA tiled/multicast/s2g, mbarrier, fence, elect, setmaxnreg, bar, ldmatrix/stmatrix and mma.sync tests with 9.0 (or 8.0). (c) Tests that launch kernels compile for `env.runnable_cuda_arch()`. It returns `sm_100a` on compute ≥ 10.0 or without a GPU, and the device's own arch below 10.0. Source-only checks keep their explicit `sm_100a`. (d) Fix the `int(env.cuda_arch()[3:6])` collection crash, mapping **each threshold on its own** (F8): `< 100` → `not has_cuda_compute(10)` (`test_cuda.py:1312`, `test_ptx_dialect.py:3132`), and `< 90` → `not has_cuda_compute(9)` (`test_ptx_dialect.py:3330`, which must keep running on sm_90). (e) Fix the `test_wgmma_rs_nt` test bug (`T.meta_var` used as a Python int). | M (mechanical across ~20 test files) | `python/tvm_tirx/testing/_pytest.py`, `tests/python/tirx/tile/cuda/conftest.py`, `tests/python/tirx/codegen/*.py`, `tests/python/tirx/tile/cuda/*.py` | On H200, `pytest tests/python/tirx/codegen tests/python/tirx/tile/cuda` with **no plugin** collects cleanly. ≥ 800 tests pass. Tests broken on every arch (A4/A5 and pre-existing test bugs) and tests needing sm_100 hardware stay at the 10.0 default, so their B200 outcome is unchanged. The fix PRs move them to 9.0. B200 is unchanged by construction: on compute ≥ 10.0 no skip is added or removed, and arch helpers return `sm_100a`. Upstream CI on B200 confirms it. |
| **A10** | **Surface the NVRTC/ptxas log** (review F1). `compile_source` throws the log away (`backend/cuda/compiler.py:50`: `binary, _log = driver(...)`). Changes: (1) keep it as `CompilationResult.log` and on the runtime module; (2) parse C75xx lines (`C7514` "wgmma.mma_async instructions are serialized", `C7517` "wait_group is injected", and the others in the C75xx range), spill lines ("bytes spill stores/loads") and stack-frame lines into structured `PerfRemark`s; (3) emit each as a Python `warnings.warn` (category `TIRxPerfWarning`); (4) add `backend_config={"cuda": {"perf_remarks": "warn" \| "error" \| "ignore"}}`, where `error` raises with the remarks listed. ptxas `-v`-only data (register counts) is optional through `ptxas: ["-v"]`. Also fix `sm90-probe/sass_check.sh`: grep `C75[0-9][0-9]`, and add a structural SASS check (per MMA chain: one `WARPGROUP.ARRIVE`, gsb0 only on the last GMMA). | S | `backend/cuda/compiler.py`, `backend/cuda/nvrtc*.py` / driver, new `backend/cuda/perf_remarks.py`, tests; `sm90-probe/sass_check.sh` | Positive control: the review's hazard kernel (an accumulator touch between two MMAs of a chain) **warns** C7514/C7517 in warn mode and **raises** in error mode. The probe GEMM and `probe_wgmma` are remark-clean. A CPU-only unit test parses canned NVRTC logs. |
| **A11** | **`setmaxnreg` launch contract** (found through A10's C7508 remark; `sm90-probe/logs/a10/setmaxnreg*.log`). Measured on H200 / CUDA 13.2: TIRx's default signature has only `__launch_bounds__(N)` (`.maxntid`). With that, ptxas **drops `setmaxnreg` entirely**: C7508 "'setmaxnreg' ignored; unable to determine register count at entry", no `USETMAXREG` in SASS, entry 8 regs. This holds for `test_ptx_setmaxnreg` and for a 384-thread warp-specialized kernel alike. With `min_blocks_per_sm` (`.minnctapersm`) or `max_registers_per_thread` (`.maxnreg`) it is kept, and the entry count is `min(255, floor8(65536 / (threads x min_blocks)))` (measured: 168 for 384x1, 80 for 384x2, 255 for 128x1) or the maxnreg. A second hazard ptxas does not diagnose: if Σ(per-warpgroup targets x 128) exceeds entry x threads, `setmaxnreg.inc` **blocks forever**. The 384-thread kernel with 40/232 at `min_blocks_per_sm=2` (entry 80) hangs on H200, killed at 60 s. Divergent (non-warpgroup-uniform) use is not diagnosed either. **Item:** (1) a lowering check: a kernel containing `tirx.ptx.setmaxnreg*` without `min_blocks_per_sm`/`max_registers_per_thread` raises a TIRx error with the source span ("ptxas would ignore it, C7508; add `KernelAttributes(min_blocks_per_sm=1)`"). This moves tirx-lite `specialize()`'s existing guard into the compiler for raw-PTX users. Not auto-emitted, because that would silently change the kernel's register policy. (2) a helper `entry_registers(threads, min_blocks_per_sm, max_registers_per_thread)` implementing the measured formula, used by (3) a budget check in tirx-lite `specialize()` (B-f), where the roles and their warps are known. | S | `backend/cuda/transform/` (or the A3 verifier pass), `backend/cuda/lang/`, tests | `test_ptx_setmaxnreg` (no attrs) raises the new error. With `min_blocks_per_sm=1` it compiles C7508-free with USETMAXREG in SASS. The helper matches ptxas's "Used N registers" for 128/256/384 threads x 1/2 blocks and for maxnreg. |
| **A3** | **Per-variant arch-floor check** (`VerifyPTXTarget`). (1) Add `InstructionEntry.floor: Callable[[mod_map], str] \| None`. Most families are constant; `cert_arch` stays the certification arch. (2) Generate per-variant floors for the ~173 families whose `cert_arch` ≥ sm_100: a script assembles each variant at sm_90a and keeps the lowest arch that passes, reusing the `PTX_CERT` renderer (`tests/.../test_ptx_dialect.py::test_ptx_all_helpers_certify`), and writes `ptx/floors_generated.py`. Overrides go where the ISA is stricter than ptxas. (3) A TIRx pass in the CUDA pipeline walks `tirx.ptx.*` calls, decodes tokens as codegen does (`engine._make_codegen`), and compares against the target arch, including the `a`/`f` suffix rule (wgmma/setmaxnreg need `sm_90a`). It raises with the TIRx source span. | M | `backend/cuda/ptx/{table,engine}.py`, new `backend/cuda/ptx/floors_generated.py`, `backend/cuda/transform/` (new pass), `tools/` (generator) | Compiling the sm_100 `fp16_bf16_gemm` for sm_90a raises a `TIRxError` naming `tcgen05.mma` and its source line **before NVRTC**. All `sm90-probe` kernels and the A2 suite still compile. A CPU-only unit test covers every family in `{sm_90a ok, sm_100a-only rejected}`. |
| **A4** | **DSMEM bulk-copy tile-pattern fix** (all archs). `remote_addresses` is allocated inside the guarded branch, so codegen fails with `Back optional access`. Hoist the alloc or bind it with `T.bind_`, as `lang/pipeline.py::_map_addr_into_cta` does. | S | `backend/cuda/tile/patterns/async_copy.py` | `test_cp_async_bulk.py::test_dsmem` 4/4 and `test_pattern_transfers.py::test_dsmem_remote_addresses` pass on H200 |
| **A5** | **mma.sync / ldmatrix tile-pattern `KeyError 'm'`** (all archs; `common.py` reads `physical["m"]` for layouts without an `m` axis). | S–M | `backend/cuda/tile/patterns/{common,mma,ldmatrix}.py` | `test_mma_sync.py` and `test_ldmatrix_stmatrix.py` fully pass on H200 |
| **A6** | **SM90 pipeline release and persistent grid sizing.** (1) `Pipeline(empty="mbar")` already works for WGMMA consumers. Add `MBarrier.arrive_multicast(stage, cta_mask)`: one `mbarrier.arrive.shared::cluster` per CTA in the multicast group through `mapa`, the hand-rolled loop in `probe_gemm.py`. Add a docstring recipe for "consumer releases after `wgmma.wait_group`". (2) Add a `max_active_clusters(kernel, cluster, smem)` host helper over `cuOccupancyMaxActiveClusters`, and use it in `ClusterPersistentScheduler2D` grid sizing (review F7: cluster 4 gives 30 clusters, not 132/4 = 33). | S | `backend/cuda/lang/pipeline.py`, `backend/cuda/lang/tile_scheduler.py`, host runtime helper | New GPU test: 2-CTA cluster with B multicast, 4-stage ring over `Pipeline` + `arrive_multicast`, exact result. For each kernel config tested, `max_active_clusters` equals the raw `cuOccupancyMaxActiveClusters` driver query. The review's 132/66/30/15 (cluster 1/2/4/8) is only an H200 regression check for the probe GEMM's own config (384 threads, about 192 KB smem). `probe_gemm.py` rewritten on it is no slower than the raw version, measured in the same step. |
| **A7** | **WGMMA authoring helpers + named barriers (not a tile pattern).** `backend/cuda/lang/wgmma.py` provides: (1) `WGMMADesc`, a Hopper smem descriptor for 128/64/32B swizzle, K- and MN-major, wrapping `cuda.wgmma_encode_matrix_descriptor` (`cpp/descriptors.py:38`), with the +2-per-32B K-step; the tcgen05 descriptor format (layout codes, version field) differs, so tirx-lite's `SmemDescriptor` cannot be reused; (2) `alloc_wgmma_AB`, the SM90 twin of `alloc_tcgen05_mma_AB`, with the same canonical swizzled layouts (to verify in the test); (3) `wgmma_chain(acc, descA, descB, k_steps, shape, dtype)`, which emits fence → noop_barrier → k MMAs with `scale_d` → commit; (4) `acc_fragment(m64nN)`, the ISA 9.7.17.5.1 (reg, lane) → (row, col) map for epilogues and FA3 softmax; (5) the `.rs` A-fragment map for register-A WGMMA; (6) `NamedBarrier(id, threads)`, which allocates `bar.sync`/`bar.arrive` ids and reserves id 0 for `cta_sync`, for warpgroup ping-pong. | M | new `backend/cuda/lang/{wgmma,named_barrier}.py`, `backend/cuda/lang/__init__.py` | GPU tests on H200: (a) one-warpgroup m64n{64,128,256}k16 bf16/fp16 and m64n128k32 e4m3 ss/rs through the helpers match torch. (b) **Chain structure** (F5) on an ss chain and an rs chain of 4 k-steps: exactly one `WARPGROUP.ARRIVE` per chain, `gsb0` only on the last GMMA, and no C75xx under A10's error mode. (c) `acc_fragment` round-trips a register tile through `stmatrix`/global. (d) Two consumer warpgroups ping-pong through `NamedBarrier`, with the order checked. (e) A **one-warpgroup FA-forward microkernel**: QKᵀ ss, softmax on `acc_fragment`, then PV rs with P from registers, one KV block loop, against torch SDPA. It is C75xx-clean, so B3's rs-form, fragment and barrier risks surface here. |
| **A8** | **Arch-aware packed-f32 helpers.** `T.cuda.fma2/mul2/add2(dst, a, b, rnd=...)` lower to `*.f32x2` when `arch ≥ sm_100` and to two scalar ops otherwise, selected from the target. tirx-lite re-exports them as `txl.idioms.*`. | S | `backend/cuda/lang/` or `cpp/` intrinsic, plus tests | `test_pattern_math` / `test_fma` SM90 variants pass; the same kernel source compiles for sm_90a and sm_100a |
| A9 | (deferred) **WGMMA tile pattern** (`T.cuda.tile.apply("wgmma...")`, fragment ownership at warpgroup scope, like `tcgen05_mma.py`). | L | `backend/cuda/tile/patterns/wgmma.py` | — |

**Decision on A9 (the WGMMA tile pattern): defer.** Reasons:

- Kernels in TIRx-kernels, and the ones the harness evolves, are written in tirx-lite with spelled-out
  PTX plus helpers. The only tile-op kernel (FA4 tileop) fails lowering even on sm_100.
- Every first kernel below works with raw `wgmma` plus the A7 helpers. The probes already prove the
  raw path.
- Revisit A9 once B1 and B3 exist, since they show which fragment manipulations recur. Revisit
  sooner if the harness moves agents to tile-level authoring.
- The deferral holds only because A7 owns the fence → MMAs → commit → wait protocol in one helper and
  A10 makes violations loud (review).

**Not needed:** new PTX entries. Sparse `wgmma.sp` is unregistered, but no planned kernel uses it.

Order inside the track: **A1 → A2 → A10 → A7 / A6 / A8** (A7 with the F5 tests) → A11 → A4, A5 → A3.
- The codegen track is ready for kernels once A1 + A2 + A10 + A7 are done (review).
- A3 is the most valuable item for agents: it turns NVRTC dumps into source-located errors. But
  nothing blocks on it.
- A3 should land before the harness evolves SM90 kernels (milestone M3).

## 3. Track B — SM90 example kernels (TIRx-kernels; one PR each)

Each kernel is a new module with `KERNEL_META["runtime_cuda_archs"] = ["sm_90a"]`. Existing sm_100
modules stay untouched. The exception is B0, where the same module can serve both archs once A8 exists.

**Targets are ratios measured in the same step**: kernel and baseline interleaved call by call, 256 MB L2
flush before every rep, median of ≥ 30, on one GPU.
- Every bench record logs the `hostname`, `clocks.sm` and the throttle reasons (`nvidia-smi --query-gpu=clocks.sm,clocks_throttle_reasons.active`,
  sampled during the run).
- Absolute TFLOPS and TB/s below are **informational only**, from one node of task 2394207_98 (`fs-mbz-gpu-[277,358,420,536]`; `baselines.log` did not record which, hence the hostname rule) on 2026-11-21. They vary by
  node and power cap: 8192³ cuBLAS ran SW-power-capped at about 1500 MHz, and gave 693–702 TFLOPS on another node
  (review F3).
- The acceptance tooling is `tirx_kernels.bench`, with interleaving added if it lacks it (B-e).

| # | Kernel | What it exercises | Baseline (informational numbers) | Gate (same-step ratio) | Depends on |
|---|---|---|---|---|---|
| **B0** (smoke test) | **RMSNorm fp16** (port `norm/rmsnorm.py`; memory-bound). It uses no WGMMA, TMA or cluster, so it is **not** one of the SM90 examples. | `f32x2` → A8 helpers; SM count from the device (132, not `SM_COUNT = 152`); registering `sm_90a`; the whole test/bench/harness flow on H200 at low risk | flashinfer `rmsnorm`, which beats `torch.rms_norm` in every row: 2.94 TB/s (hs4096 bs4113), 3.79 (hs4096 bs16384), 3.50 (hs8192 bs4113), 3.98 (hs8192 bs16384). torch: 2.3–3.2 TB/s. | ≥ 1.0x flashinfer on rows with bs ≥ 4113; ≥ 0.9x on small batches (launch-bound, about 7 us). Bandwidth fraction informational. **Result (2026-11-21, TIRx-kernels `haok/sm90-rmsnorm` `ee0c75a`): correct on all 36 CONFIGs, but 0.80–0.96x flashinfer at bs ≥ 4113 and 0.81–0.97x at bs ≤ 128, so the gate is missed.** It stays a working smoke test, and closing the gap is the **first H200 evolution task (C3)** rather than hand-tuning. | A8 (or a local scalar fallback), A2 |
| **B1** | **bf16/fp16 GEMM, NT** (warp-specialized persistent) | TMA + 2-CTA multicast, `Pipeline` with A6 release, WGMMA m64n256k16 ss through A7, `setmaxnreg` 40/232, persistent `ClusterPersistentScheduler2D` (no CLC), stmatrix + TMA-store epilogue, 132-SM wave math. Seed: `sm90-probe/probe_gemm.py` (fair timing: 0.86–0.95x on large-K squares, 0.64–0.78x on LLM shapes). | cuBLAS (`torch.matmul`); informational: 766 TFLOPS at 4096³, 786 at 8192³ (693–702 on another node), 589 at 8192x8192x1024, 636 at 4096x7168x2048 | Geomean ≥ **0.95x** cuBLAS over the `fp16_bf16_gemm` CONFIGS with **≥ 2 waves of 128x256 tiles** (≥ 264 tiles at 132 SMs). **Gated set (7 shapes x fp16/bf16):** 4096³, 8192³, 16384³, 2048x8192x1024, 2048x11008x4096, 4096x4096x11008, 4096x11008x4096. Short-K 2048x8192x1024 is the hardest of these (the probe is at 0.64x on 8192x8192x1024). **Tracked, not gated:** 1024³ (32 tiles), 2048³ (128 tiles, 0.97 waves), and the 256-tile (1.94-wave) rectangles 1024x8192x2048, 2048x4096x11008, 4096x2048x8192 and 8192x1024x2048. Small-tile and split-K variants are **out of scope** for B1. Don't aim above 1.0x: noise is 2–3%, and DeepGEMM's own sm90 bf16 GEMM averages 0.952x cuBLASLt here. Rel err ≤ 1e-2 vs fp32. **C75xx-clean under A10 error mode.** | A1, A2, A10, A6, A7; A3 nice |
| **B2** | **FP8 e4m3 block-scaled GEMM, DeepGEMM sm90 style** (1x128 activation scales, 128x128 weight scales, fp32 scales, bf16 out, NT) | fp8 WGMMA k32 (K-major only); **promotion**: WGMMA into a temporary f32 accumulator, then `acc += tmp * sa * sb` in registers every K-block; scale tiles via TMA. **B2-design (own item, F4):** two accumulators do not fit at B1's m64n256 per warpgroup (128 + 128 f32 registers before addressing; DeepGEMM keeps `accum` + `final_accum` with 232/248-register math warpgroups). Pick the N tile and the warpgroup/M split, and write down the register budget, before coding. Reuses B1's TMA, multicast, persistent scheduler and epilogue, but **not its tile**. | DeepGEMM `fp8_fp4_gemm_nt` (sm90 1D2D kernel) at the harness's commit `559d79f`, built from source (§1.2), measured with its own `tests/test_fp8_fp4.py`. On the 14 rows with m ≥ 2048 (DeepSeek-V3 projections): **936–1505 TFLOPS, geomean 1300**, which is 0.994x cuBLASLt per-tensor fp8. m=128 rows: 115–688 TFLOPS. cuBLAS fp8 rowwise `_scaled_mm` (coarser scaling): 1053 TFLOPS at 4096x7168x2048, 1272 at 4096x4096x7168, 1373 at 8192³. | Geomean ≥ **0.95x** DeepGEMM over the m ≥ 2048 1D2D rows, timed in the same step and interleaved (not DeepGEMM's kineto numbers); error `calc_diff` < 1e-3 (DeepGEMM's criterion). Small-m (decode) rows tracked, not gated. **C75xx-clean under A10 error mode** (expect "insufficient register resources" remarks during design). | B1, B2-design, A7 fp8 descriptors, A10, C1 (the DeepGEMM baseline must load) |
| **B3** | **FA3-style attention forward** (bf16, hd128, causal and non-causal, seq 1k–16k) | WGMMA **rs** form (P from registers via the A7 A-fragment map), softmax on accumulator fragments (A7 `acc_fragment`), 2 consumer warpgroups in ping-pong over named barriers, intra-warpgroup softmax/MMA overlap, `setmaxnreg` 24/240, TMA K/V ring, `ex2.approx`, `cvt.rn.bf16x2.f32` | FA3 (flashinfer `backend="fa3"`): 708 TFLOPS non-causal, 660 causal, **measured at one shape only** (b1 h16 s16k d128). cuDNN SDPA: 594–667 non-causal, 440–628 causal. FA2: 315–393. | **B3-baseline first (F9):** measure FA3 (flashinfer batch prefill, or Dao-AILab `flash_attn_3`) at **every** gate shape (s ≥ 4k, hd128, causal and non-causal, b·h as in the table) in the interleaved harness. **Then** fix the target, provisionally ≥ 0.90x of max(FA3, cuDNN). **C75xx-clean under A10 error mode.** | A1, A7 (incl. NamedBarrier + FA microkernel), A10, A6; A3 nice |

**Order: B0 (smoke) → B1 → B2 → B3.**

- B0 is a plumbing smoke test only: test gate, registration, bench, harness.
- The "couple of SM90 example kernels" are **B1 + B2**.
- **B2 comes before B3**, for four reasons:
  - PithTrain's FP8 path makes it directly useful.
  - Its DeepGEMM baseline is already built and measured.
  - Its new risk (two accumulators, register budget, QGMMA) is the territory A10 must prove.
  - It reuses B1's infrastructure.
- B3's codegen risks (rs form, fragment maps, named-barrier ping-pong) are pulled forward into A7's tests,
  the FA microkernel and the 2-warpgroup barrier test. That way they surface in track A, not at M4.

**B1 result (2026-11-21).** Gated geomean against cuBLAS is **0.986x bf16 / 0.992x fp16**, min 0.938x. It
is bit-exact, has any-M and N/K % 8 tails, passes 38/38 configs and is C75xx-clean. It needs two compiler-bug
workarounds: ISSUE-001 and ISSUE-002 in `sm90-probe/issues/`.

**B2-design result (2026-11-21).** Measured on H200: compile-only budget sweeps plus same-step timing against
DeepGEMM.

- **Register budget at 384 threads.** The pinned entry is 168, split by `setmaxnreg` as 40 / 232; 24 / 240
  spills once cluster addressing is added. Accumulators per consumer thread:

  | Tile | Accumulators (f32) | Result |
  |---|---|---|
  | 128 × 128 | 64 + 64 | clean |
  | 128 × 192 | 96 + 96 | clean |
  | 128 × 256 | 128 + 128 | **spills 236 B** |
  | 256 × 128, two m64 waves per warpgroup | 2·64 + 64 | clean (DeepGEMM's main tile) |

- **Promotion cannot be overlapped by double-buffering `tmp`.** At 128 × 128, 3 × 64 registers fit, but ptxas
  serializes every wgmma (C7514) as soon as the retired buffer is read while the next chain is in flight. The
  promotion therefore stays serial after `wait_group 0`, as in DeepGEMM, and is amortized with bigger tiles.
  The no-promotion upper bound of the same kernel matches DeepGEMM, so the promotion is the whole gap.
- **Kernel:** `tirx_kernels/gemm/fp8_blockscale_gemm_sm90.py`, configurable `(bm, bn, cluster)`. It is exact
  against DeepGEMM (`calc_diff` 0.0) on all 14 shapes.
- **Result (B2 done, TIRx-kernels `haok/sm90-fp8-gemm` `9734a79`):** gated geomean **1.001x DeepGEMM**, min 0.966x.
  Output equals DeepGEMM's (`calc_diff` 0.0); 20/20 configs pass, including tails; C75xx-clean.
  - The gap had been around the promotion, not in it. A source and SASS diff against DeepGEMM's cubin found
    the causes:
    - `#pragma unroll 8` on the k-loop: about 6–7%;
    - a warp-uniform warpgroup index via `shfl.sync`, which avoids per-wgmma R2UR: about 2–3%;
    - the ISSUE-002 workaround: about 10% on the cluster configs.
  - Decode rows (m = 128) are tracked at 0.41–0.84x; a small-M tile is future work.

**B3-baseline result (2026-11-21, `sm90-probe/logs/b3/baseline.log`).** bf16 hd128, same step, interleaved:
flashinfer FA3 against cuDNN SDPA and torch flash (FA2).

- FA3 is the faster baseline at **every** gated shape (s ≥ 4k):

  | Shape (b·h·s) | Non-causal (TFLOPS) | Causal (TFLOPS) |
  |---|---|---|
  | 4·16·4k | 693 | 617 |
  | 8·16·4k | 662 | 622 |
  | 2·16·8k | 709 | 657 |
  | 4·32·8k | 705 | 661 |
  | 1·16·16k | 722 | 718 |

  cuDNN runs 590–682. B3's target, 0.90x of max(FA3, cuDNN), is therefore **556–650 TFLOPS** depending on
  shape.
- s ≤ 2k rows are tracked only. There cuDNN is ahead (598/442 TFLOPS at 8·32·1k).
- **B3 waits until after M3** (coordinator, 2026-11-21).

Per-kernel acceptance in TIRx-kernels, all three run as 1-GPU steps:
- `python -m tirx_kernels.test --arch sm_90a --kernel <name>` passes every CONFIG.
- `python -m tirx_kernels.bench --arch sm_90a --kernel <name> --with-references` reports the ratio to
  the baseline above.
- A10 error mode (`perf_remarks: "error"`) compiles every CONFIG: no C75xx, no spills.
- The fixed `sass_check.sh` structural check passes: one ARRIVE per chain, gsb0 only on the last GMMA.

Smaller TIRx-kernels items that ride along:
- **B-a.** `tests/test_correctness.py` asserts `skipped == 0` across all kernels, so it fails on H200.
  Filter by the device arch. (S)
- **B-b.** `runner.hardware_num_sms(default=148)`: query the device and drop the default. Persistent and cluster
  grids use A6's `max_active_clusters`, not `num_sms // cluster`. (S)
- **B-c.** A WGMMA-format path in `tirx_lite/smem.py` (`KDesc`/`mma_desc(fmt="wgmma")`), or a re-export
  of A7. (S)
- **B-d.** Bump the `apache-tvm-tirx` pin to the release that contains A1, A6 and A7. Development does not wait
  for it: we develop on the `haok/sm90-plan` pair.
- **B-f.** tirx-lite `specialize()`: static `setmaxnreg` budget check (Σ role threads x regs, plus unassigned
  threads x entry, must be ≤ entry x block threads), using A11's `entry_registers`. Over budget is a silent hang. (S)
- **B-e.** `tirx_kernels.bench`: interleaved kernel/baseline timing with an L2 flush, and hostname, SM clock and throttle
  reasons in every record. (S)

## 4. Track C — TIRx-harness (evolve and benchmark on H200)

Findings come from a full read of the harness (§ file refs). Ordered by what blocks the first H200
evolution run (bf16 GEMM).

| # | Item | Size | Files | Acceptance on H200 |
|---|---|---|---|---|
| **C1** | **Benchmark stack on these nodes.** The CUDA 13.4 pins work on driver 13.2 (§1.2), so keep them. Replace the DeepGEMM wheel (glibc 2.38) with a manylinux_2_28-compatible build of `559d79f`; locally, a source build with `-std=c++20`. Mark `flash-kda` (sm_100a-only) optional. Confirm the FlashInfer CUPTI timer is what runs. The CUDA-event fallback (`benchmark_common.py:285-296`) is back-to-back with a hot L2. Make it flush L2 like the CUPTI path, or refuse to score (F10). Record hostname, SM clock and throttle reasons with every score. | S–M | `pyproject.toml`, `uv.lock`, `evolution/benchmark/flashinfer_bench_evolve/benchmark_common.py` | `uv sync --locked --group benchmark` succeeds on a compute node; `import deep_gemm` works; a `fp16_gemm_floor` run records `timer=cupti` |
| **C2** | **Target arch as data.** Add `target_archs` (and `gpu_name`) to `Task` (`evolution/preparation/declare.py:39-47`). Template the specs: "B200 (sm_100)" is hard-coded in 31 of the 32 task YAMLs. Filter tasks against the server's reported arch (kcoral reports `sm_90a`). Inject the SM count. Drop the hard-coded "Blackwell" in `evolution/prompts/render.py:55`. | M | `evolution/preparation/declare.py`, `evolution/tasks/*.yaml`, `evolution/prompts/render.py`, docs | `evolve` on an H200 server refuses B200-only tasks with a clear message and renders "H200 (sm_90a)" in SM90 task prompts |
| **C3** | **H200 task set**, one task per kernel B0–B3. **`rmsnorm_h200` is the first evolution task**: seed B0, baseline flashinfer, gate ≥ 1.0x at bs ≥ 4113 (seed: 0.80–0.96x). Then `bf16_gemm_h200` (cuBLAS; can be `fp16_gemm_floor` with the C2 arch field), `fp8_blockscale_gemm_h200` (DeepGEMM sm90 via its public API, **not** the sm100-only `_sm100_fp8_fp4_gemm_1d1d/data.py` path with `require_sm100()` and UE8M0 scales), `fa3_forward_h200` (FA3 through flashinfer's `fa3` backend, plus cuDNN). Tolerances as in B. | M | `evolution/tasks/`, `evolution/benchmark/flashinfer_bench_evolve/tasks/<new>/`, `evolution/benchmark/adapter.py` (`PACKAGED`) | Each task's baseline runs and is timed on H200; the seed kernel (B0/B1 or probe) scores. Each scoring run returns A10's C75xx/spill remarks to the agent with the score. |
| **C4** | **kcoral under Slurm, with preemption.** Run `kcoral server` as a 1-GPU `srun` step with `--gpus 0 --sandbox none --max-timeout-seconds 3600`. The default bubblewrap sandbox needs user namespaces, and the default 900 s cap is too short. A wrapper script re-resolves the running task's job ID, relaunches the server after preemption, and publishes `TIRX_BENCH_SERVER`. Add client-side retry with backoff to `evolution/remote/kcoral_remote.py:109-114` so a preempted request is re-sent, not fatal. The login node reaches compute-node ports (HTTP 200 from `fs-mbz-gpu-536`, `net.log`). | M | `evolution/remote/kcoral_remote.py`, new `scripts/slurm_kcoral.sh`, docs | Kill the server step mid-run: the evolution client waits, reconnects to the relaunched server, and finishes the run |
| **C5** | **Local mode under `srun`.** `rule_gpu.md` and `cli.py:71` tell the agent to rewrite `CUDA_VISIBLE_DEVICES`, which is wrong inside a step. `choose_device` parses only numeric `CUDA_VISIBLE_DEVICES` (`benchmark_common.py:111-113`). Accept UUIDs and trust the step's allocation. | S | `evolution/prompts/rules/rule_gpu.md`, `evolution/cli.py`, `benchmark_common.py` | Local-mode `fp16_gemm_floor` runs in a 1-GPU step without touching other agents' GPUs |
| **C6** | **Pins.** Move `apache-tvm-tirx` and `tirx-kernels` to releases containing A1/A6/A7/A8 and B0/B1 (§6). Until then, use a documented dev mode against local checkouts. | S | `pyproject.toml`, `uv.lock` | `tirx-harness` env imports the pinned pair; B1 compiles from the installed `tirx_kernels` |
| **C7** | **SM90 knowledge for the agent.** New `skills/tirx-wiki/optimization/sm90.md` covering the WGMMA fence/arrive/commit/wait protocol and `noop_barrier`, ss vs rs, register accumulators and m64nN fragment layout, register budgeting with `setmaxnreg`, cooperative vs ping-pong consumers, FA3 intra-warpgroup overlap, FP8 promotion, 132-SM persistent scheduling without CLC, persistent grids sized with `max_active_clusters`, no f32x2, and FP8/TF32 WGMMA operands being K-major only (an fp8 V or NN/TN fp8 GEMM needs an smem transpose). It also covers reading A10's C75xx remarks: what C7514/C7517 mean and the usual causes. Add B0–B3 to `kernel-catalog.md` as SM90 canonical kernels. In `tirx-profile-kernel/references/ncu.md`, use `--chip gh100`. | M | `skills/tirx-wiki/...`, `skills/tirx-profile-kernel/...` | Reviewed doc; an evolution run on `bf16_gemm_h200` cites it and does not try tcgen05 |
| **C8** | **NumSim / synccheck / racecheck for WGMMA.** Today every `tirx.ptx.wgmma_*` op is an `architecture_rejection` (`tools/frontend-rs/src/registry.rs`, `SUPPORTED_OPS.md`). Phase 1: make SM90 kernels fail soft (tool reports "WGMMA not modelled", other checks still run). Phase 2 (L), modelling WGMMA, is **out of scope** (ruling): we rely on GPU tests, compute-sanitizer and A10. | S | `tools/frontend-rs/`, `tools/src/tirx_harness/numsim/` | Phase 1: synccheck on B1 runs and reports only the WGMMA ops as unmodelled |
| C9 | Nice to have: default `dump_kernel` arch from the device (`dump_kernel.py:36`); un-skip the NumSim microtests below SM100 for non-tcgen05 cases | S | `tools/` | — |

There is **no torch.compile host-time gate** in the harness or bench suite at HEAD; the phase-1 worry
about one is moot.

## 5. Milestones and dependencies

```
A1 ──> A2 ──> A10 ─┬─> A7 (F5 tests, NamedBarrier, FA microkernel) ─┐
                   ├─> A6 (+ max_active_clusters) ──────────────────┼─> B1 ──> B2-design ──> B2 ──> B3-baseline ──> B3
                   └─> A8 ──> B0 (smoke)                            ┘
A2 ──> A4, A5;  A3 before M3
C1, C4, C5 ──> C2 ──> C3 ──> M3 (needs B1 and C6);  C7 alongside B1/B2;  C8 phase 1 before M3
```

| Milestone | Contents | Exit criterion (H200) |
|---|---|---|
| **M1: SM90 is tested and loud** | A1, A2, A10, B-a | tvm-tirx GPU suites run on H200 with no plugin, with 0 failures. The hazard kernel raises under A10 error mode. |
| **M2: first SM90 kernels** | A6, A7 (with F5 tests), A8, A4, A5; B0, B1; B-b, B-c, B-e | B0 smoke passes; B1 meets its ≥ 2-wave gate in interleaved `tirx_kernels.bench` and is C75xx-clean |
| **M3: evolve on H200** | C1, C2, C3 (rmsnorm, bf16 GEMM), C4, C5, C6, C8-phase-1, A3, C7 | An end-to-end evolution run on `bf16_gemm_h200` through kcoral in a Slurm step survives one forced preemption and produces a scored candidate |
| **M4: the second example, then attention** | B2-design, B2; B3-baseline, B3; C3 (fp8, FA3); A9 if B1/B3 show the need | B2 ≥ 0.95x DeepGEMM (same step, C75xx-clean); B3 meets the target set after B3-baseline |

**Status (2026-11-21).**
- **M3 met** (coordinator-approved substitution: run on `rmsnorm_h200`, the cheaper task).
  - The run went through the secured kcoral server, with the agent in a CPU-only Slurm step.
  - It survived a forced kill of the server step while a request was in flight.
  - Best scored candidate: **1.053x FlashInfer**, against the 0.854x seed, using 12 of 20 benchmark runs.
  - Details: `sm90-probe/logs/m3/`. Fixes for what the agent tripped on: `sm90-probe/issues/HARNESS-001`.
- **M4 met.**
  - B2: 0.997–0.998x DeepGEMM.
  - B3 (`haok/sm90-fa3` `cdf11ac`), gate metric **CUPTI kernel-only**: worst gated 0.926x
    (4×32×8k causal), geomean 0.975x over the 12 `fa3_forward_h200` rows. Non-causal runs
    1.00–1.03x, causal 0.93–0.96x; the reviewer measured 0.922–0.929x worst. Event-timed runs
    (0.957x worst) include FlashInfer's Python dispatch and are secondary.
  - Both are `perf_remarks="error"`-clean.
  - `fa3_forward_h200` (C3 part 3) is packaged, with B3 as its seed (`haok/h200-tasks` `85ab176`).
  - Open follow-up: a persistent scheduler for short causal tiles.

## 6. Version pins

### 6.1 Development pair (pinned now)

| Repo | Commit | Notes |
|---|---|---|
| tvm-tirx | **`5bdc1cc07`** = branch `haok/sm90-plan` = upstream `6569e2a18` + A1 | built in `tvm-tirx/.venv` (Z3 off, tvm-ffi 0.1.16, torch 2.15+cu132, NVRTC 13.2) |
| TIRx-kernels | **`16f05fc`** (upstream main; branch `haok/sm90-plan`) | |
| TIRx-harness | `7ec7e40` + docs (branch `haok/sm90-plan`) | `.venv` with the locked benchmark group (torch 2.15+cu134, NVRTC 13.4) + DeepGEMM `559d79f` source build |

Compatibility evidence for this pair is in §6.3 (`pair.log`). The HEAD probes in §1.2 ran on this pair.

### 6.2 Release pair (target)

- `apache-tvm-tirx` **0.1.1**, the first release containing A1. A6, A7 and A8 should land too, so
  TIRx-kernels can drop local shims.
- `tirx-kernels` **0.1.4** with B0, B1 and B-a..B-d, pinning `apache-tvm-tirx==0.1.1`.
- The harness then pins both (C6).

Until 0.1.1 exists, SM90 kernels cannot run on the released wheel. v0.1.0's `noop_barrier` crashes
codegen. That is the main reason A1 must be the first upstream PR.

### 6.3 Pair check result

`probe_pair.py` traces and NVRTC-compiles the first CONFIG of each of the 106 registered kernels, each
at its first declared arch. Result: **95 OK, 11 fail.** None of the failures is API skew between the
two repos:

| Failures | Cause |
|---|---|
| 3 | `sm_107a` kernels: NVRTC 13.2 in the tvm-tirx venv cannot target sm_107 |
| 2 | NVSHMEM is not installed (`allgather_gemm`, `gemm_reduce_scatter`) |
| 2 | `deep_ep` is not installed |
| 3 | The probe's generic `get_kernel(**CONFIGS[0])` call does not fit these kernels: `fastcu_nvfp4_gemm_gb300`, `grouped_gemm_masked_rubin`, `kda_decode_multishape` |
| 1 | `flash_attention4_tileop` fails lowering on every arch (also stated in TIRx-kernels #24) |

The phase-1 version-skew blocker is gone. On sm_90a, the stock kernels fail in ptxas as expected
(`txl.log`):
- `rmsnorm`: `mul.f32x2 requires sm_100` (fixed by A8).
- `fp16_bf16_gemm`: `tcgen05.mma` / `.cta_group::2` / `.32x32b` (needs B1).

In both cases the error is a ptxas dump, not a TIRx error, which is what A3 fixes.

## 7. How changes go upstream

This host cannot push. Each item is committed locally on a topic branch, and the coordinator pushes it
to the `haok1402/<repo>` fork and opens the PR. One PR per item, in this order:

| PR | Repo → upstream | Branch (local) | Depends on |
|---|---|---|---|
| 1 | tvm-tirx → apache/tvm-tirx | `haok/sm90-wgmma-fence` (A1) | — |
| 2 | tvm-tirx | `haok/sm90-test-gate` (A2) | merge after 1 (`test_wgmma_ss_nt` fails on H200 without A1) |
| 2b | tvm-tirx | `haok/nvrtc-perf-remarks` (A10) | — |
| 3 | tvm-tirx | `haok/dsmem-pattern-fix` (A4), `haok/mma-pattern-m` (A5) | 2 for the H200 test runs |
| 4 | tvm-tirx | `haok/sm90-wgmma-helpers` (A7, incl. NamedBarrier), `haok/sm90-pipeline-multicast` (A6, incl. `max_active_clusters`), `haok/packed-f32-portable` (A8) | 1; A7's tests use A10 |
| 5 | tvm-tirx | `haok/ptx-arch-floor` (A3) | 2 |
| 6 | TIRx-kernels → mlc-ai/TIRx-kernels | `haok/sm90-test-filter` (B-a, B-b) | — |
| 7 | TIRx-kernels | `haok/sm90-rmsnorm` (B0), `haok/sm90-gemm` (B1), later `haok/sm90-fp8-gemm` (B2), `haok/sm90-fa3` (B3) | tvm-tirx release with 1 and 4 |
| 8 | TIRx-harness → mlc-ai/TIRx-harness | `haok/h200-stack` (C1, C5), `haok/task-arch` (C2), `haok/kcoral-slurm` (C4), `haok/h200-tasks` (C3), `haok/sm90-skills` (C7), `haok/numsim-wgmma-softfail` (C8a) | C3 after 7 |

Every tvm-tirx PR description carries its H200 test log, because **upstream CI has no Hopper runner**.
A2 makes the tests runnable, but nothing upstream will run them. Asking the maintainers for a Hopper CI lane is
the user's call. Regardless, we run the A2 suite weekly as a 1-GPU step on our side, and SM90 PRs attach H200
logs. Docs and probes stay in TIRx-harness
`docs/sm90/` on `haok/sm90-plan`, as before.

## 8. Risks

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| No Hopper CI upstream: SM90 regresses silently after we land | High / Medium | A2 markers + PR logs; weekly 1-GPU A2 run on our side; an upstream lane is the user's call |
| Upstream maintainers prefer a different test-marker or helper placement (tvm-tirx `lang/` vs tirx-lite) | Medium / Low | Small PRs; A6/A7/A8 are additive; fall back to tirx-lite if declined |
| CUDA minor-version compatibility breaks: a future NVRTC emits PTX for JIT, or a feature needs a 13.4 driver | Low / High (everything stops) | tvm-tirx emits cubin (`compiler.py:44`); the stack probe joins the harness CI so a regression is caught immediately |
| B1–B3 hit register-pressure or ptxas scheduling issues (wgmma serialization, spills) that need compiler work (e.g. `noop_barrier` placement, unrolling control) | Medium / Medium | A10 makes them loud (error mode in acceptance); the structural SASS check; the B2-design register budget; A7's FA microkernel finds rs/fragment issues early |
| Perf numbers drift with node, power cap or run order | High / Medium | Same-step interleaved ratios only; clocks and throttle reasons logged with every record (F2/F3) |
| DeepGEMM / FA3 baselines drift or fail to build on glibc 2.35 | Medium / Low | Build from pinned source (done for DeepGEMM); FA3 via flashinfer JIT (works); cuDNN SDPA as second attention baseline |
| Preemption mid-benchmark corrupts or loses evolution state | High / Medium | C4 retry/relaunch; per-candidate results already persisted in the worktree |
| Shared quota on `/mnt/weka/shrd/ci` | Medium / Medium | Two venvs (~15 GB) + DeepGEMM source; no new large artifacts planned |
| NumSim stays blind to WGMMA: agents lose race/sync checking on SM90 MMA kernels | Certain until C8-phase-2 / Medium | Phase 1 soft-fail; rely on GPU correctness tests and compute-sanitizer racecheck |

## 9. Decisions (settled by the coordinator, 2026-11-21)

1. **B2 before B3**, with B3's codegen risks pulled into A7's tests.
2. **Helpers live in tvm-tirx `backend/cuda/lang`.** Development uses the `haok/sm90-plan` pair and does not
   block on 0.1.1.
3. **NumSim WGMMA modelling (C8 phase 2) is out of scope.** C8 phase 1 (soft-fail) only.
4. **Authoring is tirx-lite + raw PTX + A7 helpers.** A9 is deferred.
5. **Upstream Hopper CI lane:** the user's call, because it means contacting maintainers. We run a weekly 1-GPU
   A2 run regardless.
6. **Targets:**
   - Same-step interleaved ratios only.
   - B1: 0.95x over ≥ 2-wave shapes.
   - B2: 0.95x DeepGEMM.
   - B3: fixed after FA3 is measured at every gate shape.
   - B0: smoke.

## Appendix: evidence and how to reproduce

**Comparing source trees (F10).** The main tree's editable install (`_editable_skbc_apache_tvm_tirx`) is a
meta-path finder that ignores `PYTHONPATH`. To test a worktree:
- Disable the finder with `SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build` (or the review's
  `plugins/strip_editable.py`).
- Point `PYTHONPATH` at `<worktree>/python` and `TVM_TIRX_LIBRARY_PATH` at the main build's
  `site-packages/tvm_tirx/lib`. This is valid only while the change is Python-only.
- Pass `-p report_tvm_tirx` so pytest prints the imported `tvm_tirx.__file__` in its header.
- `sm90-probe/run_a1.sh` and `run_a2.sh` do all of this.

New in this session (`kernel-evolve/sm90-probe/`):

| Log (`logs/plan/`) | What | Command |
|---|---|---|
| `build.log` | tvm-tirx HEAD + A1 build | appendix of `sm90-codegen.md`, after `git submodule update --init --recursive` |
| `wgmma.log`, `gemm.log`, `cluster_pdl.log`, `txl.log` | phase-1 probes on HEAD | `CPUS=16 ./run.sh logs/plan/reverify.log ./reverify_head.sh` |
| `upstream_hopper.log/.xml` | Hopper/DSMEM/TMA suites, gate removed by `plugins/sm90_unskip.py` (test pins unchanged, so 55 sm_100a-only binaries fail with NO_BINARY_FOR_GPU) | same script |
| `upstream_gate_asis.log` | the same Hopper/TMA files with the upstream gate: 281/281 skipped | same script |
| `upstream_broad.log/.xml`, `upstream_broad_classified.txt` | codegen + tile/cuda suites, `sm_100a` pins → `sm_90a` (`upstream_sm90_head/`), tcgen05/blackwell files excluded | same script |
| `stack.log` | harness benchmark stack (CUDA 13.4) on driver 13.2 | `probe_stack.py` in `TIRx-harness/.venv` |
| `baselines.log` | cuBLAS bf16/fp8, flashinfer rmsnorm, SDPA cuDNN/FA2, FA3 (flashinfer) | `probe_baselines.py` |
| `deepgemm_build.log`, `deepgemm.log` | DeepGEMM `559d79f` source build (c++20) + its own sm90 fp8/bf16 benchmarks | see `deepgemm.log` header |
| `pair.log` | version-pair compile check of all registered kernels | `probe_pair.py` |
| `net.log` | login node → compute node HTTP reachability (kcoral) | — |
