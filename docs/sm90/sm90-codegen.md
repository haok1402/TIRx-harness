# SM90 (H200) code generation in tvm-tirx: evidence report (phase 1)

Date: 2026-11-02. Author: agent sm90-tirx-design.
Repos: tvm-tirx `3ad801cb` (+ branch `haok/sm90-design`), TIRx-kernels `716d8a3`, TIRx-harness `31aafff`.
Hardware: 1x NVIDIA H200 (driver 595.91.07, CUDA 13.2.2) as Slurm steps of PithTrain task 2394207_103.
Evidence lives in `kernel-evolve/sm90-probe/` (probe sources, generated `.cu` in `out/`, PTX/cubin/SASS
and ptxas reports in `out/sass/`, raw logs and JUnit XML in `logs/`).

## TL;DR

* **Yes, tvm-tirx can generate the SM90 code fast kernels need, today, through raw PTX.** The PTX
  instruction table already covers the full sm_90a set: WGMMA (all dtypes, ss/rs, f32/f16 acc), TMA
  load/store/multicast, mbarrier with tx-count, cluster barriers, DSMEM (`mapa`, `ld.shared::cluster`,
  bulk smem-to-peer copies, `st.async`), `setmaxnreg`, `elect.sync`, `griddepcontrol`, `ldmatrix`/`stmatrix`.
  Every one of them compiled for `sm_90a` through `pipeline="tirx"` and ran correctly on an H200.
* **Performance proof:** a ~130-line raw-TIRx warp-specialized bf16 GEMM (TMA + 2-CTA multicast +
  4-stage mbarrier ring + setmaxnreg 40/232 + WGMMA m64n256k16 with `wait_group 1`) is bit-identical to
  cuBLAS and runs at **0.87–0.98x cuBLAS** (659–754 TFLOPS) with clean SASS (back-to-back HGMMA, no
  serialization warning, no spills). The codegen is not the bottleneck.
* **One real compiler bug blocked WGMMA**: the accumulator fence `T.cuda.wgmma.noop_barrier` crashed
  codegen and fenced a copy rather than the register. **Fixed** (8 lines, `tvm-tirx@haok/sm90-design` 2909603d).
  It survived because **tvm-tirx's test harness skips every GPU codegen/tile test on non-sm_100 devices**,
  so the Hopper suite had never run on Hopper.
* **The gap is in the layers above PTX, not in codegen:** no WGMMA tile pattern; tirx-lite descriptors,
  pipelines, schedulers and idioms are tcgen05/CLC-specific; packed-`f32x2` math (75 of 152 kernel files)
  is sm_100-only; no target-arch check, so wrong-arch instructions fail late in NVRTC. The rest are
  hardware differences (TMEM, 2-SM MMA, CLC, block-scaled/NVFP4 MMA), so **existing sm_100 kernels must
  be re-tiled for SM90, not recompiled**. None of the 105 registered kernels targets sm_90a.
* Host cost is fine: 4.1 us per call for a tiny TIRx kernel vs 4.7 us for `torch.add_`, and 9.7 us
  for a GEMM call including two tensor-map encodes vs 11.1 us for `torch.matmul`
  (Python → tvm-ffi, H200 node).

## 1. What the TIRx pipeline emits today, and what the sm_100 kernels rely on

`tvm_tirx.compile(mod, target=Target({"kind": "cuda", "arch": ...}), pipeline="tirx")` lowers TIRx to a
CUDA C++ translation unit in which every hardware instruction is an inline-`asm` helper generated
from the PTX instruction table (`python/tvm_tirx/backend/cuda/ptx/table.py`, 11.7k lines, PTX ISA
9.x transcribed per instruction family), then compiles it with NVRTC (default) or nvcc for the
`backend_config.arch` (`backend/cuda/backend_config.py`). The host side is a tvm-ffi packed function
that encodes TMA descriptors (`runtime.cuTensorMapEncodeTiled`) and launches with
`cuLaunchKernelEx` attributes (`backend/cuda/launch/table.py`: cluster dims, preferred cluster,
PDL, ...). Kernels in TIRx-kernels are written in tirx-lite (`txl`), a traced PTX-level DSL over the
same IR: almost every instruction is spelled literally (`txl.ptx["tcgen05.mma.cta_group::2.kind::f16"]`),
plus a thin layer of helpers (`txl.Pipeline`, `txl.specialize`, `txl.smem_pool`, `SmemDescriptor`,
`ClusterLaunchControlScheduler`, `txl.idioms`).

Architecture is therefore a property of (a) which PTX the kernel author spells and (b) a few helper
layers. There is **no arch-specific lowering pass**: the same pipeline emits sm_90a code as long as
the kernel only spells sm_90a instructions.

Blackwell-only features used by the current TIRx-kernels kernels (static regex scan of the 152 kernel source files,
`sm90-probe/logs/kernel_feature_scan.txt`):

```
files scanned: 152 (TIRx-kernels 716d8a3, tirx_kernels/**/*.py > 3000 chars, excl. tests/bench/tirx_lite)
  60  tcgen05.mma (UMMA)
  58  TMEM (tcgen05.ld/st)
  59  tcgen05.alloc
  21  tcgen05.cp (smem->TMEM, scales)
  21  2-CTA MMA/TMA (cta_group::2)
   7  CLC (cluster launch control)
  18  block-scaled MMA (mxf8/nvfp4)
  75  packed f32x2 math
  28  preferred_cluster launch attr
  62  TMA (cp.async.bulk.tensor)
  24  TMA multicast
  34  clusters
  38  setmaxnreg / regs=
  31  griddepcontrol (PDL)
  22  ldmatrix/stmatrix
  10  mma.sync (SM80 MMA)
   0  wgmma (SM90 MMA)
```

None of the 105 registered kernels lists `sm_90a` in `KERNEL_META["runtime_cuda_archs"]`
(62 x `[sm_100a, sm_103a, sm_107a]`, 21 x `+sm_110a`, the rest single Blackwell parts). Several hard-code
Blackwell SM counts (e.g. `norm/rmsnorm.py: SM_COUNT = 152`; H200 has 132).

## 2. Feature-by-feature status on sm_90a

Legend: **Supported** = compiled for sm_90a by the `tirx` pipeline, ran on H200, and is numerically
correct (SASS checked where noted). **Partial** = works through raw PTX but a helper or tile layer is
missing or broken. **Missing** = cannot be expressed today. Paths: `ptx/table.py` =
`tvm-tirx/python/tvm_tirx/backend/cuda/ptx/table.py`. Probes are in `sm90-probe/`, and test runs are
tvm-tirx's own suites run on H200 with the sm_100 blanket skip removed (`plugins/sm90_unskip.py`).

| Feature | Status | Evidence (H200 run) | tvm-tirx code path |
|---|---|---|---|
| WGMMA bf16 / fp16, f32 acc, ss form, m64n{128,256}k16 | **Supported** | `probe_wgmma.py` 4 cases, max abs err 1.5e-5 (`logs/wgmma.log`); SASS `HGMMA.64x256x16.F32.BF16` | `ptx/table.py:10907` (`wgmma_{f16,bf16}_{ss,rs}`) |
| WGMMA fp16 with f16 acc | **Supported** | `wgmma_ss_fp16_f16acc` ok (err 2.7e-2 vs fp32 ref, expected for f16 acc) | `wgmma_f16_f16acc_*` |
| WGMMA fp8 e4m3 / e5m2 (k32, K-major only), f32 and f16 acc | **Supported** | `wgmma_ss_e4m3/e5m2_f32acc`, `e4m3_f16acc` ok; SASS `QGMMA` | `ptx/table.py:11041` (`wgmma_fp8_*`) |
| WGMMA A from registers (`.rs`) | **Supported** | `wgmma_rs_bf16`, `wgmma_rs_e4m3` ok | same entries, `form="rs"` |
| `wgmma.fence` / `commit_group` / `wait_group N` (incl. overlap `wait_group 1`) | **Supported** | `_staged` case, GEMM probe; SASS `WARPGROUP.ARRIVE` / `DEPBAR.LE gsb0` | `ptx/table.py:11193` |
| WGMMA accumulator fence (`T.cuda.wgmma.noop_barrier`) | **Fixed here** | crashed codegen before (`test_wgmma_ss_nt` FAIL), passes after 2909603d | `backend/cuda/cpp/asm.py` |
| WGMMA descriptor encode | **Supported** (raw); txl wrapper **partial** | probes use `T.cuda.wgmma.encode_matrix_descriptor` + 32B K-steps | `backend/cuda/cpp/descriptors.py:38`; txl `smem.py` only has tcgen05 format |
| WGMMA tile pattern (`T.cuda.tile.apply` / `Tx.gemm`-style) | **Missing** | no `patterns/wgmma.py`; dispatch in `tile/patterns/__init__.py` raises `No explicit tile pattern` | `backend/cuda/tile/patterns/` |
| `wgmma.mma_async.sp` (sparse) | **Missing** | not registered (table comment) | `ptx/table.py:10902` |
| TMA tiled load g2s, 2–5D, swizzle 32/64/128B, L2 hints | **Supported** | upstream `test_hopper` unicast 12/12, swizzle 9/9; `test_cp_async_bulk_tensor` (sm_90a copy) 44 pass; GEMM probe `UTMALDG.2D` | `ptx/table.py` `cp_async_bulk_tensor_g2s_*`, `tile/patterns/tma.py` |
| TMA store s2g + bulk_group | **Supported** | `test_cp_async_bulk_tensor_shared_to_global` 3/3 | `cp_async_bulk_tensor_s2g` (`ptx/table.py:8950`) |
| TMA multicast (`.multicast::cluster`) | **Supported** | upstream multicast1 5/5, multicast2 3/3; GEMM probe 2-CTA B multicast, SASS `UTMALDG.2D.MULTICAST` | `ptx/table.py:4103` |
| TMA `tile::gather4`, `.cta_group::N`, `.shared::cta` dst, interleaved/packed-fp4 maps | **N/A on SM90** (hardware) | ptxas "requires sm_100" (8+8 tests); 26 encodes rejected by the H200 driver | — |
| Tensor-map encode on host (`cuTensorMapEncodeTiled`) | **Supported** | 14/14 validation tests; per call it is part of the host function (see host cost) | `src/backend/cuda/runtime/cuda_device_api.cc` |
| mbarrier init/arrive/expect_tx/try_wait.parity, `fence.mbarrier_init.release.cluster` | **Supported** | GEMM probe 4-stage full/empty ring; SASS `SYNCS.ARRIVE.TRANS64`, `SYNCS.PHASECHK.TRANS64` | `ptx/table.py:9502, 9951`; `T.cuda.mbarrier_wait` (`cpp/asm.py:292`) |
| `fence.proxy.async` | **Supported** | `test_fence_proxy_async` pass; probes | `ptx/table.py:9513` |
| Thread-block clusters (`LaunchConfig(cluster=...)`, `barrier.cluster`, `cluster_ctaid`) | **Supported** | GEMM probe, `dsmem_*`; SASS `UCGABAR_ARV/WAIT` | `launch/table.py` CLUSTER_DIMENSION |
| `preferred_cluster` launch attribute | **Supported (accepted)** | `preferred_cluster` probe launches and runs on H200 | `launch/table.py` |
| DSMEM: `mapa` + `ld.shared::cluster` | **Supported** | `dsmem_ld` ok; `test_mapa` pass | `ptx/table.py:8464` |
| DSMEM: `cp.async.bulk.shared::cluster.shared::cta` (raw PTX) | **Supported** | `dsmem_bulk` ok; SASS `UBLKCP.S.S` | `ptx/table.py:8711` |
| DSMEM bulk-copy **tile pattern** | **Broken (all archs)** | `test_dsmem[*]` 4/5 fail: `Back optional access` | `tile/patterns/async_copy.py` |
| `st.async ... mbarrier::complete_tx` to a peer CTA | **Supported** | `dsmem_st_async` ok; SASS `STAS`. Spell it `T.ptx.st_async.shared__cluster...`: `T.ptx["st.async..."]` does not parse (dotted mnemonic) | `ptx/table.py` `st_async` |
| Remote mbarrier arrive (consumer release across CTAs) | **Supported** | GEMM probe (`mapa.shared::cluster.u32` + `mbarrier.arrive.shared::cluster`) | `lang/pipeline.py::_mbarrier_arrive_remote` |
| Warp specialization + `setmaxnreg.inc/dec` | **Supported** | GEMM probe (40/232), SASS `USETMAXREG.DEALLOC 0x28` / `TRY_ALLOC 0xe8`; `test_ptx_setmaxnreg` 2/2; txl `specialize(regs=)` probe ok | `ptx/table.py:11611`; txl `specialize.py` |
| `elect.sync`, named barriers `bar.sync/arrive id,cnt` | **Supported** | `test_ptx_elect_sync`, `test_bar_*` pass | `ptx/table.py:9424` |
| PDL: `griddepcontrol.wait/launch_dependents` + `programmatic_stream_serialization` | **Supported** | `pdl` probe correct 20/20; SASS `ACQBULK` / `PREEXIT` (overlap itself not measured) | `ptx/table.py:9749`, `launch/table.py:100` |
| `ldmatrix` / `stmatrix` (raw PTX) | **Supported** | `test_stmatrix_sync_aligned` 2/2, `test_ptx_stmatrix` 6/6, `_noncontiguous` 6/6 | `ptx/table.py:10681, 10768` |
| ldmatrix/stmatrix/mma.sync **tile patterns** | **Partial (all archs)** | `test_ldmatrix_stmatrix` / `test_mma_sync`: most pass, 30 + 6 parametrizations fail with lowering `KeyError 'm'` (`tile/patterns/common.py` `physical["m"]`); pattern lowering never reads the target, so this is not SM90-specific | `tile/patterns/{ldmatrix,mma,common}.py` |
| `mma.sync` m16n8k16 (SM80 MMA) | **Supported** | `test_mma_sync` k16 cases pass | `ptx/table.py` `mma*` |
| Packed `add/mul/fma.f32x2` | **N/A on SM90** (hardware) | ptxas "Feature 'mul.f32x2' requires sm_100" (`test_cuda`, `test_pattern_math`, `test_fma`) | — |
| tcgen05 MMA / TMEM / `tcgen05.cp` / 2-CTA / CLC | **N/A on SM90** (hardware) | sm_100 GEMM compiled for sm_90a fails **at NVRTC** with `.kind::f16`, `.cta_group::2`, `.32x32b`, `.is_canceled` errors (`probe_txl.py gemm_sm100_on_sm90`); 252 pattern tests fail on `tcgen05.alloc` | — |
| Arch check of instructions vs target | **Missing** | errors come from NVRTC on the generated C++, not from TIRx (above) | `ptx/engine.py` (no check) |
| tirx-lite generic helpers (`smem_pool`, `Pipeline(full="tma", empty="mbar")`, `PipelineState`, `specialize`) | **Supported** | `probe_txl.py txl_pipeline_specialize`: warp-specialized TMA ring, exact copy on H200 | `TIRx-kernels/tirx_kernels/tirx_lite/` |
| tirx-lite tcgen05-coupled helpers (`SmemDescriptor`/`KDesc`/`mma_desc`, `TCGen05Bar`, `ClusterLaunchControlScheduler`, `idioms.mma_chain`) | **Missing for SM90** | encode tcgen05 descriptors / emit `tcgen05.*` | `tirx_lite/{smem,pipeline,scheduler,idioms}.py` |
| Host launch cost (tvm-ffi call from Python, no stream context) | **OK** | tiny kernel 4.1 us/call vs `torch.add_` 4.7; GEMM incl. 2 tensor-map encodes 9.7 us/call vs `torch.matmul` 11.1 (`logs/batch2.log`, `batch3.log`) | `backend/cuda/host.py` |

**End-to-end performance evidence** (`probe_gemm.py`, `logs/gemm.log`): a ~130-line raw-TIRx warp-specialized
bf16 GEMM (CTA 128x256x64, 4 stages, 1 producer + 2 consumer warpgroups, 2-CTA cluster with B multicast,
register epilogue, non-persistent) is bit-identical to cuBLAS and reaches:

| M x N x K | TIRx probe | cuBLAS (torch.matmul) | ratio |
|---|---|---|---|
| 4096^3 | 689 TFLOPS (199 us) | 790 TFLOPS | 0.87 |
| 2048 x 4096 x 8192 | 754 TFLOPS | 806 TFLOPS | 0.94 |
| 8192^3 | 659 TFLOPS | 669 TFLOPS | 0.98 |
| 4096^3, no cluster/multicast | 704 TFLOPS | 790 TFLOPS | 0.89 |

ptxas: 168 registers at launch (setmaxnreg moves consumers to 232), 0 spills, no "wgmma serialized"
warning, 4 HGMMA back-to-back per k-block with one `gsb0` wait. The remaining gap at 4096 is the expected
one: no persistent scheduler, no TMA-store or stmatrix epilogue, and no producer/consumer epilogue overlap.
The codegen is not limiting.

Summary of upstream suites run on H200 (`logs/upstream_broad_classified.txt`):
1083 pass / 359 fail / 52 skip (tile + codegen, sm_100a pins rewritten to sm_90a, tcgen05 files
excluded). Of the 359: 317 sm_100-only instructions correctly rejected by ptxas, 26 interleaved/packed-fp4
TMA descriptors rejected by the driver, 11 arch-independent tile-pattern bugs, and 5 test artifacts.


## 3. Gaps, their size, and the minimal tvm-tirx changes

Sizes: **S** = under ~100 lines plus tests, **M** = a few hundred lines in one module, **L** = a new
subsystem or a rewrite of an existing kernel family. Gaps fall into four kinds: hard hardware limits,
compiler bugs, missing helpers, and tests/infrastructure.

### 3.1 Hard hardware limits (no compiler change can close them)

| sm_100 feature | SM90 replacement the kernels must use instead | Impact |
|---|---|---|
| `tcgen05.mma` + TMEM accumulators (`tcgen05.alloc/ld/st`) | `wgmma.mma_async` with accumulators in **registers** (64xN per warpgroup, N/2 f32 regs/thread). The epilogue reads registers directly (no `tcgen05.ld`). | Every MMA kernel (60/152 files) must be **re-tiled**, not ported: register pressure caps tiles at 64xN per warpgroup (128x256 per 2 consumer WGs in the probe GEMM, 168 regs at launch, raised to 232 by setmaxnreg). The MMA warp + epilogue warp split of sm_100 becomes "consumer warpgroups both MMA and epilogue" (or ping-pong). |
| `cta_group::2` (2-SM UMMA, paired TMA) | Cluster multicast TMA (proven, `UTMALDG.2D.MULTICAST`) with per-CTA WGMMA. No shared-operand MMA across SMs. | 21 files. Tile shapes and smem budgets change. |
| CLC (`clusterlaunchcontrol.try_cancel`) | Static persistent scheduler (`ClusterPersistentScheduler2D` exists and is arch-neutral) or a global atomic tile counter. | 7 files. |
| `tcgen05.cp` + block-scaled MMA (`kind::mxf8f6f4`, `mxf4nvf4`) | No native block scaling. FP8 with per-128 block scales is done DeepGEMM-sm90 style: WGMMA into a temporary f32 accumulator, then FFMA `acc += tmp * scale` in registers. NVFP4/MXFP4 has no tensor-core path on SM90. | 18 + 21 files. NVFP4 kernels have no SM90 analogue at all. |
| Packed `add/mul/fma.f32x2` (sm_100+) | Two scalar FP32 ops. Hopper's FP32 issue rate makes this neutral for throughput, but it is a source change everywhere. | 75 files: every norm/decode/quant kernel. ptxas rejects them at sm_90a (`test_pattern_math`, `test_cuda`). |
| `tile::gather4` TMA, `.cta_group::N` on TMA, `.shared::cta` TMA destination | Plain tiled TMA with per-row issues, or `cp.async`. | MSA/sparse kernels. |

### 3.2 Compiler bugs found and their fixes

| Bug | Evidence | Fix | Size |
|---|---|---|---|
| `T.cuda.wgmma.noop_barrier` (the WGMMA accumulator fence) crashed codegen (`'TensorLoad' object has no attribute 'dtype'`), so `test_wgmma_ss_nt` failed. Its helpers also took the register **by value**, so the empty `asm volatile("" : "+f"(reg))` fenced a copy rather than the accumulator. | `logs/upstream_hopper.xml` (before), `logs/wgmma.log` (after) | **Done** on `tvm-tirx@haok/sm90-design` 2909603d: `reg.expr_ty().dtype`, and `float&`/`uint32_t&` signatures (`python/tvm_tirx/backend/cuda/cpp/asm.py`). | S (8 lines) |
| DSMEM bulk-copy tile pattern (`T.cuda.tile.apply("cp.async.bulk.shared::cluster.shared::cta...", remote_cta_id=...)`) fails in codegen: `Back optional access` on a `tirx.alloc_tensor` emitted inside the branch. This is arch-independent (it would fail on B200 too). The raw-PTX path works on H200 (probe `dsmem_bulk`). | `logs/upstream_hopper.xml` (`test_dsmem[*]`) | Hoist the `remote_addresses` local out of the guarded region in `backend/cuda/tile/patterns/async_copy.py`, or bind it with `T.bind_` like `lang/pipeline.py::_map_addr_into_cta`. | S |
| `PTX table cert_arch` is never checked against the compile target. An sm_100-only instruction compiled for sm_90a surfaces only as an NVRTC/ptxas error in the generated C++, with no TIRx source location. | `probe_txl.py gemm_sm100_on_sm90` | Add a per-variant minimum arch to `InstructionEntry`. `cert_arch` cannot be reused because it is the family maximum, e.g. `cp_async_bulk_s2c` is `sm_103a` although its base form is sm_90. Check it in `ptx/engine.py` at lowering, raising a TIRx error with the kernel's source span. | M (table annotation is mechanical: ISA "Target ISA notes" per entry) |
| `test_wgmma_rs_nt` itself is broken (`'Var' object cannot be interpreted as an integer`, the `T.meta_var` count is used as a Python int). | `logs/upstream_hopper.xml` | Test fix; the instruction path itself works (`probe_wgmma` rs cases). | S |

### 3.3 Missing helper layers (the real work for productive SM90 kernels)

1. **WGMMA tile pattern** (`backend/cuda/tile/patterns/wgmma.py`, M). sm_100 has `tcgen05_mma.py`, and
   SM80 has `mma.py` with a fragment-ownership model (`ctx.region(...).owner(...)`). WGMMA needs the
   same thing at warpgroup scope: D fragment `(*group, M/64, N, 64, n)` with the ISA 9.7.17.5.1 lane
   map (`row = 16*warp + lane/4 + 8*(r%4>=2)`, `col = 8*(r/4) + 2*(lane%4) + r%2`), A-in-registers
   fragments for `.rs`, automatic `wgmma.fence` / `noop_barrier` around the chain, and K accumulation
   through `scale_d`. Raw `T.ptx["wgmma..."]` already works today (the probes), so this is about
   ergonomics and safety, not capability.
2. **WGMMA shared-memory descriptors in tirx-lite** (`TIRx-kernels/tirx_kernels/tirx_lite/smem.py`, S–M).
   `SmemDescriptor`, `KDesc` and `KTileView.mma_desc/encode` all call
   `T.cuda.tcgen05.encode_matrix_descriptor`. That is the tcgen05 format (version=1, layout codes
   2/4/6 for 128/64/32B), which differs from the Hopper GMMA descriptor (layout codes 1/2/3, no version
   field). tvm-tirx already has `cuda.wgmma_encode_matrix_descriptor`
   (`backend/cuda/cpp/descriptors.py:38`); the txl wrappers need a `fmt="wgmma"` path. The K-step
   arithmetic (+2 per 32 B inside a 128B-swizzle atom) is identical and was validated by the probes.
3. **tirx-lite Pipeline barrier kinds** (S). `txl.Pipeline(empty="tcgen05")` commits with `tcgen05.commit`.
   On SM90 consumers release with a plain (local or `mapa`-remote) `mbarrier.arrive` after
   `wgmma.wait_group`. `empty="mbar"` already works on sm_90a (probe `txl_pipeline_specialize`). Add a
   multicast-aware consumer release (`arrive` on every CTA in the multicast group, as in the probe GEMM)
   so kernels do not hand-roll the `mapa` loop.
4. **Packed-f32x2 portability idioms** (S in tirx-lite, but 75 call-site files). Add `txl.idioms.fma2/mul2/add2`
   that emit `*.f32x2` on sm_100+ and two scalar ops on sm_90, chosen from the entry's resolved arch.
   Ported kernels then call the idiom. This is the single largest blocker for porting the non-MMA
   kernels (norms, decode, quantization) to H200.
6. **Version skew TIRx-kernels HEAD vs tvm-tirx HEAD** (not SM90-specific, but it blocks porting).
   `norm/rmsnorm.py` does not trace with tvm-tirx `3ad801cb` even for **sm_100a**
   (`txl.Cast(goff, "int64")`: `prim.Cast` now takes `(type, value)`; `probe_txl.py
   rmsnorm_trace_sm100_control`). The fp16/bf16 GEMM traces fine. Phase 2 must pin a tvm-tirx commit
   that TIRx-kernels traces with, or fix the call sites.
5. **SM90 scheduler defaults** (S). `ClusterPersistentScheduler2D` is arch-neutral. Kernels choose CLC
   by name, so SM90 variants must choose the persistent scheduler. SM counts must come from the
   device (132 on H200), not constants.

### 3.4 Tests and infrastructure

* `T.cuda.cluster_sync()` lowers to the default release/acquire `barrier.cluster`, which emits
  `MEMBAR.ALL.GPU` + `CCTL.IVALL` in SASS (`out/sass/dsmem_*.sass`). That is fine at kernel start and
  end; in loops use `barrier.cluster.arrive.relaxed` as the sm_100 GEMM does.

* `tvm_tirx/testing/_pytest.py::codegen_collection_modifyitems` and `tests/python/tirx/tile/cuda/conftest.py`
  **skip every GPU test unless the device is sm_100+**, including the Hopper suite (`test_hopper.py`) and
  all TMA tests. Upstream CI has therefore never exercised SM90 at runtime, which is how the WGMMA
  fence bug survived. Minimal change (S): gate per test (`has_cuda_compute(9)` markers already exist
  on the Hopper tests) and keep the blanket skip only for files that need sm_100. Many TMA tests also
  pin `arch="sm_100a"` in helpers (`_compile_module`, `_context`). Parameterizing them over the
  detected arch is mechanical (this report ran them via a sed copy, `sm90-probe/upstream_sm90/`).
* Build: on the ifm nodes (gcc 11.4) the prebuilt `mlc-z3-static` wheel does not link (`undefined
  reference to std::__cxx11::basic_string::_M_replace_cold`; it needs a newer libstdc++). Build with
  `-DTVM_TIRX_USE_Z3=OFF` (used here) or provide gcc >= 13. Command in the appendix.


## 4. What this means for phase 2

1. **Seed SM90 kernels can be written now** in raw TIRx / tirx-lite `txl.ptx[...]`; `sm90-probe/probe_gemm.py`
   is a ready seed (bf16 GEMM at 0.87–0.98x cuBLAS) with obvious evolution headroom (persistent scheduler,
   stmatrix + TMA-store epilogue, ping-pong consumers, cluster shape). The probe set gives correct WGMMA
   recipes for fp8 (e4m3/e5m2, K-major) and register-A operands, needed by attention and KDA-style kernels.
2. **Small enabling changes worth landing before evolution starts** (in order of value per effort):
   (a) the noop_barrier fix (done); (b) let the tvm-tirx GPU test gate run sm_90 tests on H200, so
   regressions are caught; (c) tirx-lite: WGMMA descriptor format in `SmemDescriptor`/`KDesc`, an
   `fma2/mul2/add2` idiom with an sm_90 scalar fallback, and an mbarrier-based multicast-aware consumer
   release; (d) a per-variant arch floor check in the PTX engine, so agents get a TIRx-level error
   instead of an NVRTC dump; (e) the WGMMA tile pattern (largest, optional for raw-PTX kernels).
3. **Pin versions.** TIRx-kernels HEAD does not trace rmsnorm against tvm-tirx HEAD, even for sm_100a.
   The harness needs one pinned, mutually compatible (tvm-tirx, TIRx-kernels) pair per target.
4. **Benchmark plumbing is cheap on H200:** compile + run of each probe took seconds in a 1-GPU
   Slurm step, and host per-call cost via tvm-ffi is already at or below torch's. The phase-2 design must
   still deal with preemption (per-item results, resume) and the torch.compile host-time gate.

## Appendix: how to reproduce

All heavy work ran as 1-GPU Slurm steps of the running PithTrain task (`sm90-probe/run.sh`, which
looks up the job id with `squeue -j 2394207 -t RUNNING -h -o %A`).

```bash
# build tvm-tirx (gcc 11 on ifm nodes: static Z3 does not link -> Z3 off)
mkdir -p /mnt/weka/shrd/ci/haok/tvm-tirx/.venv && ln -s /mnt/weka/shrd/ci/haok/tvm-tirx/.venv tvm-tirx/.venv
srun --jobid=<job> -N1 --gres=gpu:1 --cpus-per-task=32 --mem=96G bash -lc '
  export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda CMAKE_BUILD_PARALLEL_LEVEL=32
  export CMAKE_ARGS="-DTVM_TIRX_USE_CUDA=/usr/local/cuda -DTVM_TIRX_USE_LLVM=ON -DTVM_TIRX_USE_Z3=OFF -DTVM_TIRX_USE_RUST=OFF"
  cd tvm-tirx && uv sync --no-default-groups --group test --group cuda'

cd sm90-probe
PY=$HOME/kernel-evolve/tvm-tirx/.venv/bin/python
./run.sh logs/wgmma.log  "$PY probe_wgmma.py"          # 10 WGMMA cases
./run.sh logs/gemm.log   "$PY probe_gemm.py"           # WS GEMM vs cuBLAS
./run.sh logs/batch3.log "$PY probe_cluster_pdl.py"    # DSMEM, PDL, preferred cluster, host cost
./run.sh logs/txl.log    "$PY probe_txl.py"            # tirx-lite on sm_90a
./run.sh logs/sass.log   "./sass_check.sh"             # PTX/cubin/SASS + ptxas -v into out/sass/
# upstream suites on H200: -p sm90_unskip (plugins/) removes the sm_100 blanket skip
./run.sh logs/upstream_hopper_tests.log "cd ~/kernel-evolve/tvm-tirx && $PY -m pytest -p no:xdist -p sm90_unskip \
   tests/python/tirx/codegen/test_hopper.py tests/python/tirx/codegen/test_dsmem.py \
   tests/python/tirx/tile/cuda/test_cp_async_bulk.py tests/python/tirx/tile/cuda/test_cp_async_bulk_tensor.py"
```

Files: `probe_*.py` (sources), `out/*.cu` (generated CUDA per case), `out/sass/*.{ptx,sass,ptxas.txt}`,
`logs/*.log` (JSON-lines results), `logs/*.xml` (JUnit), `logs/upstream_broad_classified.txt`,
`logs/kernel_feature_scan.txt`, `upstream_sm90/` (test copies with sm_100a pins rewritten).
