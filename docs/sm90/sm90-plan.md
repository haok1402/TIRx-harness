# SM90 (H200) support for TIRx: plan

Date: 2026-11-21. Author: agent sm90-plan. Builds on `docs/sm90-codegen.md` (phase 1, 2026-11-02).
Hardware for every number here: 1x H200 (driver 595.91.07, CUDA 13.2 toolkit on the node), 1-GPU Slurm
steps of PithTrain task 2394207_98. New evidence: `sm90-probe/logs/plan/` (index in the appendix).

## 0. Verdict: is PTX codegen the missing piece?

**No, and HEAD shows the same.** The PTX table already emits every sm_90a instruction fast kernels
need. On tvm-tirx HEAD (`6569e2a18` plus the fix below) the phase-1 probes reproduce unchanged:

- WGMMA bf16/fp16/fp8, ss and rs forms: 10/10 cases correct (`wgmma.log`).
- Warp-specialized GEMM: bit-identical to cuBLAS at 0.87 to 0.96x of its speed (`gemm.log`).
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
| Raw-TIRx WS GEMM at 0.87–0.98x cuBLAS | **Holds**: 0.87 / 0.94 / 0.96 (4096³, 2048x4096x8192, 8192³) | `gemm.log` |
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
| **A1** | **WGMMA accumulator fence fix.** `wgmma_noop_barrier` crashed codegen (`'TensorLoad' object has no attribute 'dtype'`) and fenced a by-value copy. Already done as `5bdc1cc07`. Add a source-level test: the helper takes `float&`/`uint32_t&`, and the asm sits on the accumulator. | S (done + test) | `backend/cuda/cpp/asm.py`, `tests/python/tirx/codegen/test_hopper.py` | `test_hopper.py::test_wgmma_ss_nt` passes; new `test_wgmma_noop_barrier_by_reference` (CPU source check) passes; `probe_wgmma.py` 10/10 |
| **A2** | **Let SM90 tests run on SM90.** (a) Replace the blanket "compute ≥ 10.0" skip in `codegen_collection_modifyitems` and `tile/cuda/conftest.py` with a per-test minimum: new marker `pytest.mark.cuda_min_compute(9, 0)`. Unmarked GPU tests keep requiring 10.0, so B200 behaviour is unchanged. (b) Mark the Hopper, DSMEM, TMA tiled/multicast/s2g, mbarrier, fence, elect, setmaxnreg, bar, ldmatrix/stmatrix and mma.sync tests with 9.0 (or 8.0). (c) Replace hard-coded `"sm_100a"` in test helpers (`_compile_module`, `_context`, `TARGET`) with the detected arch, falling back to sm_100a without a GPU. (d) Fix the `int(env.cuda_arch()[3:6])` collection crash with `env.has_cuda_compute(10)`. (e) Fix the `test_wgmma_rs_nt` test bug (`T.meta_var` used as a Python int). | M (mechanical across ~20 test files) | `python/tvm_tirx/testing/_pytest.py`, `tests/python/tirx/tile/cuda/conftest.py`, `tests/python/tirx/codegen/*.py`, `tests/python/tirx/tile/cuda/*.py` | On H200, `pytest tests/python/tirx/codegen tests/python/tirx/tile/cuda` with **no plugin** collects cleanly. ≥ 800 tests pass. The only failures left are A4/A5, marked `xfail(strict=True)` until those land. On B200 the pass count is unchanged (reviewer or CI). |
| **A3** | **Per-variant arch-floor check** (`VerifyPTXTarget`). (1) Add `InstructionEntry.floor: Callable[[mod_map], str] \| None`. Most families are constant; `cert_arch` stays the certification arch. (2) Generate per-variant floors for the ~173 families whose `cert_arch` ≥ sm_100: a script assembles each variant at sm_90a and keeps the lowest arch that passes, reusing the `PTX_CERT` renderer (`tests/.../test_ptx_dialect.py::test_ptx_all_helpers_certify`), and writes `ptx/floors_generated.py`. Overrides go where the ISA is stricter than ptxas. (3) A TIRx pass in the CUDA pipeline walks `tirx.ptx.*` calls, decodes tokens as codegen does (`engine._make_codegen`), and compares against the target arch, including the `a`/`f` suffix rule (wgmma/setmaxnreg need `sm_90a`). It raises with the TIRx source span. | M | `backend/cuda/ptx/{table,engine}.py`, new `backend/cuda/ptx/floors_generated.py`, `backend/cuda/transform/` (new pass), `tools/` (generator) | Compiling the sm_100 `fp16_bf16_gemm` for sm_90a raises a `TIRxError` naming `tcgen05.mma` and its source line **before NVRTC**. All `sm90-probe` kernels and the A2 suite still compile. A CPU-only unit test covers every family in `{sm_90a ok, sm_100a-only rejected}`. |
| **A4** | **DSMEM bulk-copy tile-pattern fix** (all archs). `remote_addresses` is allocated inside the guarded branch, so codegen fails with `Back optional access`. Hoist the alloc or bind it with `T.bind_`, as `lang/pipeline.py::_map_addr_into_cta` does. | S | `backend/cuda/tile/patterns/async_copy.py` | `test_cp_async_bulk.py::test_dsmem` 4/4 and `test_pattern_transfers.py::test_dsmem_remote_addresses` pass on H200 |
| **A5** | **mma.sync / ldmatrix tile-pattern `KeyError 'm'`** (all archs; `common.py` reads `physical["m"]` for layouts without an `m` axis). | S–M | `backend/cuda/tile/patterns/{common,mma,ldmatrix}.py` | `test_mma_sync.py` and `test_ldmatrix_stmatrix.py` fully pass on H200 |
| **A6** | **SM90 pipeline release.** `Pipeline(empty="mbar")` already works for WGMMA consumers. Add `MBarrier.arrive_multicast(stage, cta_mask)`: one `mbarrier.arrive.shared::cluster` per CTA in the multicast group through `mapa`, the hand-rolled loop in `probe_gemm.py`. Add a docstring recipe for "consumer releases after `wgmma.wait_group`". | S | `backend/cuda/lang/pipeline.py` | New GPU test: 2-CTA cluster with B multicast, 4-stage ring over `Pipeline` + `arrive_multicast`, exact result. `probe_gemm.py` rewritten on it stays at ≥ 0.87x cuBLAS. |
| **A7** | **WGMMA authoring helpers (not a tile pattern).** `backend/cuda/lang/wgmma.py` provides: (1) `WGMMADesc`, a Hopper smem descriptor for 128/64/32B swizzle, K- and MN-major, wrapping `cuda.wgmma_encode_matrix_descriptor` (`cpp/descriptors.py:38`), with the +2-per-32B K-step; the tcgen05 descriptor format (layout codes, version field) differs, so tirx-lite's `SmemDescriptor` cannot be reused; (2) `alloc_wgmma_AB`, the SM90 twin of `alloc_tcgen05_mma_AB`, with the same canonical swizzled layouts (to verify in the test); (3) `wgmma_chain(acc, descA, descB, k_steps, shape, dtype)`, which emits fence → noop_barrier → k MMAs with `scale_d` → commit; (4) `acc_fragment(m64nN)`, the ISA 9.7.17.5.1 (reg, lane) → (row, col) map for epilogues and FA3 softmax; (5) the `.rs` A-fragment map for register-A WGMMA. | M | new `backend/cuda/lang/wgmma.py`, `backend/cuda/lang/__init__.py` | GPU tests on H200: (a) one-warpgroup m64n{64,128,256}k16 bf16/fp16 and m64n128k32 e4m3 ss/rs through the helpers match torch; (b) `acc_fragment` round-trips a register tile through `stmatrix`/global; (c) `probe_wgmma.py` cases re-expressed with the helpers are byte-identical in SASS HGMMA count |
| **A8** | **Arch-aware packed-f32 helpers.** `T.cuda.fma2/mul2/add2(dst, a, b, rnd=...)` lower to `*.f32x2` when `arch ≥ sm_100` and to two scalar ops otherwise, selected from the target. tirx-lite re-exports them as `txl.idioms.*`. | S | `backend/cuda/lang/` or `cpp/` intrinsic, plus tests | `test_pattern_math` / `test_fma` SM90 variants pass; the same kernel source compiles for sm_90a and sm_100a |
| A9 | (deferred) **WGMMA tile pattern** (`T.cuda.tile.apply("wgmma...")`, fragment ownership at warpgroup scope, like `tcgen05_mma.py`). | L | `backend/cuda/tile/patterns/wgmma.py` | — |

**Decision on A9 (the WGMMA tile pattern): defer.** Reasons:

- Kernels in TIRx-kernels, and the ones the harness evolves, are written in tirx-lite with spelled-out
  PTX plus helpers. The only tile-op kernel (FA4 tileop) fails lowering even on sm_100.
- Every first kernel below works with raw `wgmma` plus the A7 helpers. The probes already prove the
  raw path.
- Revisit A9 once B1 and B3 exist, since they show which fragment manipulations recur. Revisit
  sooner if the harness moves agents to tile-level authoring.

**Not needed:** new PTX entries. Sparse `wgmma.sp` is unregistered, but no planned kernel uses it.

Order inside the track: A1 → A2 (with A4/A5 xfails) → A6, A7, A8 in parallel → A4, A5 → A3.
- A3 is the most valuable item for agents: it turns NVRTC dumps into source-located errors. But
  nothing blocks on it.
- A3 should land before the harness evolves SM90 kernels (milestone M3).

## 3. Track B — SM90 example kernels (TIRx-kernels; one PR each)

Each kernel is a new module with `KERNEL_META["runtime_cuda_archs"] = ["sm_90a"]`. Existing sm_100
modules stay untouched. The exception is B0, where the same module can serve both archs once A8 exists.

**Baselines were measured on this H200 today** (`baselines.log`). Timing: CUDA events, 256 MB L2 flush
before every rep, median of 30.

| # | Kernel | What it exercises | Baseline on H200 (measured) | Target | Depends on |
|---|---|---|---|---|---|
| **B0** | **RMSNorm fp16** (port `norm/rmsnorm.py`; memory-bound) | `f32x2` → A8 helpers; SM count from the device (132, not `SM_COUNT = 152`); registering `sm_90a`; the whole test/bench/harness flow on H200 at low risk | flashinfer `rmsnorm`, which beats `torch.rms_norm` in every row: 2.94 TB/s (hs4096 bs4113), 3.79 (hs4096 bs16384), 3.50 (hs8192 bs4113), 3.98 (hs8192 bs16384). torch: 2.3–3.2 TB/s. | ≥ 1.0x flashinfer on rows with bs ≥ 4113; ≥ 0.9x on small batches (launch-bound, about 7 us); ≥ 4.0 TB/s at hs8192 bs16384 (83% of 4.8 TB/s) | A8 (or a local scalar fallback), A2 |
| **B1** | **bf16/fp16 GEMM, NT** (warp-specialized persistent) | TMA + 2-CTA multicast, `Pipeline` with A6 release, WGMMA m64n256k16 ss through A7, `setmaxnreg` 40/232, persistent `ClusterPersistentScheduler2D` (no CLC), stmatrix + TMA-store epilogue, 132-SM wave math. Seed: `sm90-probe/probe_gemm.py` (0.87–0.96x today). | cuBLAS (`torch.matmul`): 766 TFLOPS at 4096³, 786 at 8192³, 789 at 2048x4096x8192, 589 at 8192x8192x1024, 636 at 4096x7168x2048, 689 at 16384x2112x7168 | Geomean ≥ **0.95x** cuBLAS over the existing `fp16_bf16_gemm` CONFIGS (square 1k–16k plus its LLM rows), no row below 0.85x; stretch ≥ 1.0x. For calibration, DeepGEMM's own sm90 bf16 GEMM averages **0.952x** cuBLASLt on this H200 (`deepgemm.log`), so 1.0x means beating DeepGEMM. Rel err ≤ 1e-2 vs fp32. | A1 (hard), A6, A7, A2; A3 nice |
| **B2** | **FP8 e4m3 block-scaled GEMM, DeepGEMM sm90 style** (1x128 activation scales, 128x128 weight scales, fp32 scales, bf16 out, NT) | fp8 WGMMA k32 (K-major only); **promotion**: WGMMA into a temporary f32 accumulator, then `acc += tmp * sa * sb` in registers every K-block (2x accumulator register pressure); scale tiles via TMA; persistent and multicast as in B1 | DeepGEMM `fp8_fp4_gemm_nt` (sm90 1D2D kernel) at the harness's commit `559d79f`, built from source (§1.2), measured with its own `tests/test_fp8_fp4.py`. On the 14 rows with m ≥ 2048 (DeepSeek-V3 projections): **936–1505 TFLOPS, geomean 1300**, which is 0.994x cuBLASLt per-tensor fp8. m=128 rows: 115–688 TFLOPS. cuBLAS fp8 rowwise `_scaled_mm` (coarser scaling): 1053 TFLOPS at 4096x7168x2048, 1272 at 4096x4096x7168, 1373 at 8192³. | Geomean ≥ **0.95x** DeepGEMM over the m ≥ 2048 1D2D rows (about 1235 TFLOPS); stretch 1.0x; error `calc_diff` < 1e-3 (DeepGEMM's criterion). Small-m (decode) rows tracked, not gated. | B1 done (shares the skeleton), A7 fp8 descriptors, C1 (the DeepGEMM baseline must load) |
| **B3** | **FA3-style attention forward** (bf16, hd128, causal and non-causal, seq 1k–16k) | WGMMA **rs** form (P from registers via the A7 A-fragment map), softmax on accumulator fragments (A7 `acc_fragment`), 2 consumer warpgroups in ping-pong over named barriers, intra-warpgroup softmax/MMA overlap, `setmaxnreg` 24/240, TMA K/V ring, `ex2.approx`, `cvt.rn.bf16x2.f32` | FA3 (flashinfer `backend="fa3"`): **708** TFLOPS non-causal, **660** causal (b1 h16 s16k d128). cuDNN SDPA: 594–667 non-causal, 440–628 causal. FA2: 315–393. | ≥ **0.90x** of max(FA3, cuDNN) on s ≥ 4k hd128, both causal modes; stretch 1.0x | A1, A7 (fragment maps), A6; A3 nice |

**Recommended order: B0 → B1 → (B2 ∥ B3).**

- B0 is cheap and proves the full path: test gate, registration, bench, harness.
- B1 is the template the other two copy.
- The coordinator asked for "a couple": **B0 + B1 are the minimum.** B2 and B3 are the next two.
  They are where SM90-specific codegen issues are most likely to show up: register pressure with two
  accumulators, rs-form fragment layout, named-barrier ping-pong.

Per-kernel acceptance in TIRx-kernels, all three run as 1-GPU steps:
- `python -m tirx_kernels.test --arch sm_90a --kernel <name>` passes every CONFIG.
- `python -m tirx_kernels.bench --arch sm_90a --kernel <name> --with-references` reports the ratio to
  the baseline above.
- SASS check (`sass_check.sh`): no spills, no "wgmma serialized" ptxas warning, HGMMA back-to-back.

Smaller TIRx-kernels items that ride along:
- **B-a.** `tests/test_correctness.py` asserts `skipped == 0` across all kernels, so it fails on H200.
  Filter by the device arch. (S)
- **B-b.** `runner.hardware_num_sms(default=148)`: query the device and drop the default. (S)
- **B-c.** A WGMMA-format path in `tirx_lite/smem.py` (`KDesc`/`mma_desc(fmt="wgmma")`), or a re-export
  of A7. (S)
- **B-d.** Bump the `apache-tvm-tirx` pin to the release that contains A1, A6 and A7.

## 4. Track C — TIRx-harness (evolve and benchmark on H200)

Findings come from a full read of the harness (§ file refs). Ordered by what blocks the first H200
evolution run (bf16 GEMM).

| # | Item | Size | Files | Acceptance on H200 |
|---|---|---|---|---|
| **C1** | **Benchmark stack on these nodes.** The CUDA 13.4 pins work on driver 13.2 (§1.2), so keep them. Replace the DeepGEMM wheel (glibc 2.38) with a manylinux_2_28-compatible build of `559d79f`; locally, a source build with `-std=c++20`. Mark `flash-kda` (sm_100a-only) optional. Confirm the FlashInfer CUPTI timer, not the silent CUDA-event fallback, is what runs (`benchmark_common.py:271-294`), and make the fallback loud. | S–M | `pyproject.toml`, `uv.lock`, `evolution/benchmark/flashinfer_bench_evolve/benchmark_common.py` | `uv sync --locked --group benchmark` succeeds on a compute node; `import deep_gemm` works; a `fp16_gemm_floor` run records `timer=cupti` |
| **C2** | **Target arch as data.** Add `target_archs` (and `gpu_name`) to `Task` (`evolution/preparation/declare.py:39-47`). Template the specs: "B200 (sm_100)" is hard-coded in all 32 task YAMLs. Filter tasks against the server's reported arch (kcoral reports `sm_90a`). Inject the SM count. Drop the hard-coded "Blackwell" in `evolution/prompts/render.py:55`. | M | `evolution/preparation/declare.py`, `evolution/tasks/*.yaml`, `evolution/prompts/render.py`, docs | `evolve` on an H200 server refuses B200-only tasks with a clear message and renders "H200 (sm_90a)" in SM90 task prompts |
| **C3** | **H200 task set**, one task per kernel B0–B3: `rmsnorm_h200` (flashinfer), `bf16_gemm_h200` (cuBLAS; can be `fp16_gemm_floor` with the C2 arch field), `fp8_blockscale_gemm_h200` (DeepGEMM sm90 via its public API, **not** the sm100-only `_sm100_fp8_fp4_gemm_1d1d/data.py` path with `require_sm100()` and UE8M0 scales), `fa3_forward_h200` (FA3 through flashinfer's `fa3` backend, plus cuDNN). Tolerances as in B. | M | `evolution/tasks/`, `evolution/benchmark/flashinfer_bench_evolve/tasks/<new>/`, `evolution/benchmark/adapter.py` (`PACKAGED`) | Each task's baseline runs and is timed on H200; the seed kernel (B0/B1 or probe) scores |
| **C4** | **kcoral under Slurm, with preemption.** Run `kcoral server` as a 1-GPU `srun` step with `--gpus 0 --sandbox none --max-timeout-seconds 3600`. The default bubblewrap sandbox needs user namespaces, and the default 900 s cap is too short. A wrapper script re-resolves the running task's job ID, relaunches the server after preemption, and publishes `TIRX_BENCH_SERVER`. Add client-side retry with backoff to `evolution/remote/kcoral_remote.py:109-114` so a preempted request is re-sent, not fatal. The login node reaches compute-node ports (HTTP 200 from `fs-mbz-gpu-536`, `net.log`). | M | `evolution/remote/kcoral_remote.py`, new `scripts/slurm_kcoral.sh`, docs | Kill the server step mid-run: the evolution client waits, reconnects to the relaunched server, and finishes the run |
| **C5** | **Local mode under `srun`.** `rule_gpu.md` and `cli.py:71` tell the agent to rewrite `CUDA_VISIBLE_DEVICES`, which is wrong inside a step. `choose_device` parses only numeric `CUDA_VISIBLE_DEVICES` (`benchmark_common.py:111-113`). Accept UUIDs and trust the step's allocation. | S | `evolution/prompts/rules/rule_gpu.md`, `evolution/cli.py`, `benchmark_common.py` | Local-mode `fp16_gemm_floor` runs in a 1-GPU step without touching other agents' GPUs |
| **C6** | **Pins.** Move `apache-tvm-tirx` and `tirx-kernels` to releases containing A1/A6/A7/A8 and B0/B1 (§6). Until then, use a documented dev mode against local checkouts. | S | `pyproject.toml`, `uv.lock` | `tirx-harness` env imports the pinned pair; B1 compiles from the installed `tirx_kernels` |
| **C7** | **SM90 knowledge for the agent.** New `skills/tirx-wiki/optimization/sm90.md` covering the WGMMA fence/arrive/commit/wait protocol and `noop_barrier`, ss vs rs, register accumulators and m64nN fragment layout, register budgeting with `setmaxnreg`, cooperative vs ping-pong consumers, FA3 intra-warpgroup overlap, FP8 promotion, 132-SM persistent scheduling without CLC, and no f32x2. Add B0–B3 to `kernel-catalog.md` as SM90 canonical kernels. In `tirx-profile-kernel/references/ncu.md`, use `--chip gh100`. | M | `skills/tirx-wiki/...`, `skills/tirx-profile-kernel/...` | Reviewed doc; an evolution run on `bf16_gemm_h200` cites it and does not try tcgen05 |
| **C8** | **NumSim / synccheck / racecheck for WGMMA.** Today every `tirx.ptx.wgmma_*` op is an `architecture_rejection` (`tools/frontend-rs/src/registry.rs`, `SUPPORTED_OPS.md`). Phase 1: make SM90 kernels fail soft (tool reports "WGMMA not modelled", other checks still run). Phase 2 (L): model WGMMA (async-proxy smem reads, register accumulators, commit/wait-group semantics, Hopper descriptor) plus an sm_90a arch profile. | S, then L | `tools/frontend-rs/`, `tools/src/tirx_harness/numsim/` | Phase 1: synccheck on B1 runs and reports only the WGMMA ops as unmodelled |
| C9 | Nice to have: default `dump_kernel` arch from the device (`dump_kernel.py:36`); un-skip the NumSim microtests below SM100 for non-tcgen05 cases | S | `tools/` | — |

There is **no torch.compile host-time gate** in the harness or bench suite at HEAD; the phase-1 worry
about one is moot.

## 5. Milestones and dependencies

```
A1 ─┬─> A2 ──> (A4, A5)
    ├─> A6 ─┐
    ├─> A7 ─┼─> B1 ─┬─> B2 (needs C1 DeepGEMM)
    └─> A8 ─┴─> B0  └─> B3
A3 (independent; before M3)
C1, C4, C5 ──> C2 ──> C3 ──> M3 (needs B0/B1 and C6)
C7 alongside B1/B3; C8 phase 1 before M3, phase 2 later
```

| Milestone | Contents | Exit criterion (H200) |
|---|---|---|
| **M1: SM90 is tested** | A1, A2 (A4/A5 as strict xfail), B-a | tvm-tirx GPU suites run on H200 with no plugin: ≥ 800 pass, 0 unexpected failures |
| **M2: first SM90 kernels** | A6, A7, A8, A4, A5; B0, B1; B-b, B-c | B0 and B1 meet their targets in `tirx_kernels.bench --arch sm_90a` |
| **M3: evolve on H200** | C1, C2, C3 (rmsnorm, bf16 GEMM), C4, C5, C6, C8-phase-1, A3, C7 | An end-to-end evolution run on `bf16_gemm_h200` through kcoral in a Slurm step survives one forced preemption and produces a scored candidate |
| **M4: hard kernels** | B2, B3; C3 (fp8, FA3); A9 if B1/B3 show the need; C8 phase 2 optional | B2 ≥ 0.95x DeepGEMM; B3 ≥ 0.90x of max(FA3, cuDNN) |

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
| 2 | tvm-tirx | `haok/sm90-test-gate` (A2) | 1 (or carries A1's test as xfail) |
| 3 | tvm-tirx | `haok/dsmem-pattern-fix` (A4), `haok/mma-pattern-m` (A5) | 2 for the H200 test runs |
| 4 | tvm-tirx | `haok/sm90-pipeline-multicast` (A6), `haok/sm90-wgmma-helpers` (A7), `haok/packed-f32-portable` (A8) | 1 |
| 5 | tvm-tirx | `haok/ptx-arch-floor` (A3) | 2 |
| 6 | TIRx-kernels → mlc-ai/TIRx-kernels | `haok/sm90-test-filter` (B-a, B-b) | — |
| 7 | TIRx-kernels | `haok/sm90-rmsnorm` (B0), `haok/sm90-gemm` (B1), later `haok/sm90-fp8-gemm` (B2), `haok/sm90-fa3` (B3) | tvm-tirx release with 1 and 4 |
| 8 | TIRx-harness → mlc-ai/TIRx-harness | `haok/h200-stack` (C1, C5), `haok/task-arch` (C2), `haok/kcoral-slurm` (C4), `haok/h200-tasks` (C3), `haok/sm90-skills` (C7), `haok/numsim-wgmma-softfail` (C8a) | C3 after 7 |

Every tvm-tirx PR description carries its H200 test log, because **upstream CI has no Hopper runner**.
A2 makes the tests runnable, but nothing upstream will run them. We should ask the maintainers about a
Hopper CI lane, or at least agree that SM90 PRs attach H200 logs. Docs and probes stay in TIRx-harness
`docs/sm90/` on `haok/sm90-plan`, as before.

## 8. Risks

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| No Hopper CI upstream: SM90 regresses silently after we land | High / Medium | A2 markers + PR logs; ask maintainers for an H100/H200 lane; run the A2 suite in our harness CI (weekly 1-GPU step) |
| Upstream maintainers prefer a different test-marker or helper placement (tvm-tirx `lang/` vs tirx-lite) | Medium / Low | Small PRs; A6/A7/A8 are additive; fall back to tirx-lite if declined |
| CUDA minor-version compatibility breaks: a future NVRTC emits PTX for JIT, or a feature needs a 13.4 driver | Low / High (everything stops) | tvm-tirx emits cubin (`compiler.py:44`); the stack probe joins the harness CI so a regression is caught immediately |
| B2/B3 hit register-pressure / ptxas scheduling issues (wgmma serialization, spills) that need compiler work (e.g. `noop_barrier` placement, unrolling control) | Medium / Medium | SASS check per change; these are exactly the "more codegen issues" phase 2 is meant to find; budget time in M4 |
| DeepGEMM / FA3 baselines drift or fail to build on glibc 2.35 | Medium / Low | Build from pinned source (done for DeepGEMM); FA3 via flashinfer JIT (works); cuDNN SDPA as second attention baseline |
| Preemption mid-benchmark corrupts or loses evolution state | High / Medium | C4 retry/relaunch; per-candidate results already persisted in the worktree |
| Shared quota on `/mnt/weka/shrd/ci` | Medium / Medium | Two venvs (~15 GB) + DeepGEMM source; no new large artifacts planned |
| NumSim stays blind to WGMMA: agents lose race/sync checking on SM90 MMA kernels | Certain until C8-phase-2 / Medium | Phase 1 soft-fail; rely on GPU correctness tests and compute-sanitizer racecheck |

## 9. Open questions (need a decision)

1. **Kernel priority after B1:** FP8 block-scaled GEMM (B2) or FA3 forward (B3) first? PithTrain's FP8
   path would argue for B2. B3 surfaces more new codegen territory (rs form, fragment layouts).
2. **Helper placement:** WGMMA helpers and packed-f32 helpers in **tvm-tirx `backend/cuda/lang`**
   (consistent with v0.1.0 moving pipeline/scheduler there, but gated on a tvm-tirx release) or in
   **tirx-lite** (iterates faster)? The plan assumes tvm-tirx, with a temporary tirx-lite copy until 0.1.1.
3. **Is NumSim WGMMA modelling (C8 phase 2, L) in scope**, or do we accept GPU-only verification for SM90
   MMA kernels?
4. **Authoring level for agents on SM90:** OK to stay with tirx-lite + raw PTX + A7 helpers (A9 deferred)?
5. **Upstream CI:** do we ask apache/tvm-tirx for a Hopper runner, or keep SM90 testing in our harness?
6. **Targets:** are 0.95x cuBLAS (B1; DeepGEMM bf16 itself is 0.952x), 0.95x DeepGEMM (B2), 0.90x of
   max(FA3, cuDNN) (B3) and 1.0x flashinfer (B0) the right bars, or should evolution aim higher from the start?

## Appendix: evidence and how to reproduce

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
