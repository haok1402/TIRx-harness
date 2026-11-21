# KERNEL-001: first synccheck / racecheck findings on the SM90 kernels (B1, B2)

These are not compiler issues: they concern the kernels. They were found once C8 phase 1 let the checkers run
on WGMMA kernels (TIRx-harness `haok/wgmma-fail-soft` `5feb943`). Script: `sm90-probe/c8_negative.py` /
`c8_b2.py`; run on a CPU step.

## 1. B1 `fp16_bf16_gemm_sm90`: one thread releases a stage for its whole warpgroup (synccheck error)

At M = N = 256, K = 512 (8 k-blocks over 3 stages, so stages are reused), synccheck reports:

> fixed Mbarrier protocol rejected … warp:5 … mbarrier … generation 0 wait … can be overtaken by generation 1
> completion; next mutation … warp:0

- **The pattern.** In B1, only consumer thread `ct == 0` (warp 4 of WG1, warp 8 of WG2) arrives on the empty
  barrier, on behalf of the warpgroup's four warps, after its own `wgmma.wait`.
- **The hazard.** If the other three warps are not converged with it, warp 4 can release stage *s*. The
  producer can then refill it and complete the full barrier's next generation before warp 5 has executed its
  `full.wait(s, phase)` on the old generation.
- **Why it may be safe today.** In hardware, WGMMA's `.sync.aligned` warpgroup instructions plausibly converge
  the warpgroup. Phase 1 does not model WGMMA, so the checker cannot see that convergence. B1 also passed the
  reviewer's 112 stress configs and the sanitizers.
- **Robust regardless:** release per warp, from lane 0 of each consumer warp with empty count = 8 per CTA.
  DeepGEMM does this, and so does B2, where synccheck is clean. Suggested follow-up: apply it to B1 and
  re-measure, since B2 measured per-warp release as neutral.

## 2. B2 `fp8_blockscale_gemm_sm90`: generic read of TMA-loaded scales before the stage is refilled (racecheck)

In both checked configs (256×128 and 128×192), racecheck reports a `data_race` on the A-scale tile `S_s`:

> unordered read/write … warp:4 … read shared … and warp:0 … write shared …; memory ordering failure: no
> generic->async proxy bridge from shared::cta to shared::cluster

- **The pattern.** Consumers read `S_s[stage, …]` with generic-proxy `ld.shared`, then arrive on the empty
  barrier. The producer waits on it and issues the next TMA write (async proxy) into the same bytes.
- **What racecheck wants.** Its model requires a `fence.proxy.async` between the generic read and the async
  write.
- **Comparison.** DeepGEMM uses the same pattern (`ld_shared(smem_sfa[stage])`, then `empty_barrier_arrive`)
  with no fence, and B2 is bit-exact under stress.
- **Open question for review.** Does the PTX memory model require the proxy fence for this
  write-after-read (mbarrier release/acquire between them), or is racecheck stricter than the
  architecture?
- **If required:** one `fence.proxy.async.shared::cta` per consumer warp before the empty arrive, once per
  k-block.

## Resolution (2026-11-21)

### #1: a checker false positive; per-warp release kept anyway

The reviewer judged the finding a false positive: WGMMA's `.sync.aligned` warpgroup instructions converge
the warpgroup, which phase 1 cannot see. Per-warp release costs nothing, so it stays.

TIRx-kernels `haok/sm90-gemm` `289b650` makes lane 0 of every consumer warp release, with empty count 8 per CTA.

| Check | Result |
|---|---|
| synccheck, K=512 and K=1024 | `review` with only the WGMMA advisory (K=512 was an error) |
| `tirx_kernels.test` | 38/38 |
| error-mode compile | clean; SASS chains `[4, 4]` |
| gated geomean vs cuBLAS | **0.989x bf16 / 0.993x fp16** (was 0.986 / 0.992) |

### #2: the fence is required by the memory model. Added; it costs ~0.3–0.4%

**The model** (PTX ISA, the copy bundled in tirx-wiki `references/manuals/ptx_isa.rst`):

- **§8.6 Proxies.** "A *proxy fence* is required to synchronize memory operations across different *proxies*."
  `.async.bulk` operations (TMA) access memory through the async proxy. An ordinary `ld.shared` is the generic
  proxy.
- **§8.9.5 Causality Order.** X precedes Y in *proxy-preserved base causality order* only if X and Y use the
  generic proxy; or use the same proxy from the same thread block; or are aliases with an alias proxy fence on
  the path. Causality order, which decides whether two conflicting accesses race, is built from this order.
- Our X is the consumer's generic `ld.shared` of `S_s[stage]`. Y is the producer's next TMA write to the same
  bytes, in the async proxy.
  - The mbarrier arrive (release) → wait (acquire) → TMA issue chain puts X before Y in *base* causality order.
  - None of the three conditions holds, so X does not precede Y in *proxy-preserved* order.
  - The two are not *morally strong* either (§8.7 requires the same proxy).
  - So under the model they race: the read may observe the new TMA data.
- **§9.7.10.28.2 Async Proxy.** "Accessing the same memory location across multiple proxies needs a cross-proxy
  fence. For the async proxy, `fence.proxy.async` should be used." The implicit generic-async fence it describes
  follows the *completion* of the bulk copy: that covers RAW (TMA write, then generic read), not this WAR.
- **§9.7.15.4 membar/fence.** A bi-directional proxy fence "take[s] effect within a single thread" and composes
  with other synchronization. So each reading thread must execute it before the release that orders it.

**The CUDA Programming Guide** (`references/manuals/cuda_programming_guide.rst`):

- **§3.2.2.3.1 Async Thread and Async Proxy:** "Async thread operating in async proxy … *Prior and subsequent*
  normal loads and stores to the same address are not guaranteed to maintain their ordering. A proxy fence is
  required to synchronize them across the different proxies." Our consumer read is exactly a *prior normal load*
  relative to the later TMA.
- **§4.14.2.1** (cluster launch control, thread-block use case) handles the same write-after-read reuse: after
  reading the async-proxy-written result, "Release read of result to the async proxy:
  `fence_proxy_async_generic_sync_restrict(sem_release, space_shared, scope_cluster)`".

**Practice, which does not follow the model here:**

- **CUTLASS 4.2.1** `PipelineTmaAsync::consumer_release` (`include/cutlass/pipeline/sm90_pipeline.hpp`) is a bare
  `empty_barrier.arrive`, with no proxy fence.
- Its FP8 blockwise-scaling mainloop
  (`gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp`) reads the possibly
  TMA-loaded scales with a generic `copy(tCsSFA(..., read_stage), tCrSFA)`, then calls
  `pipeline.consumer_release` without a fence.
- DeepGEMM does the same: `ld_shared(smem_sfa[stage])`, then `empty_barrier_arrive`.
- Both rely on current hardware: the generic loads have completed, with their values in registers, before the
  warp's mbarrier arrive retires. The model doesn't guarantee this, and the guide says async behaviour "may
  change in the future".

**So racecheck is right, not stricter than the model.** It flags exactly what the formal model calls a race.
Production code tolerates it on today's hardware.

**Two orderings are required, not one.** The reviewer found this independently, and it converges with
`5a68eef`:

1. **Generic read → async write, within each lane: the proxy fence.** This is the cross-proxy ordering
   argued above.
2. **Lane *i*'s read → lane 0's arrive, across lanes: `__syncwarp`.**
   - §9.7.15.4: a bi-directional proxy fence "take[s] effect within a single thread". Only lane 0 arrives
     (the release).
   - So lane 1's fence orders lane 1's read before lane 1's *later* operations, but not before lane 0's
     arrive.
   - A fence alone therefore leaves a cross-lane hole. The warp sync closes it: every lane's fenced read →
     `__syncwarp` → lane 0's release arrive → producer acquire → TMA write.
   - An alternative is to have every lane arrive (count 32 × warps). That costs more mbarrier traffic and
     was not measured.

**Fix and cost** (TIRx-kernels `haok/sm90-fp8-gemm` `5a68eef`; comment and PR text naming both orderings in
`193a7a5`):
- Every consumer thread runs `fence.proxy.async.shared::cta` after its last read of the stage (ordering 1).
- Then `__syncwarp` (ordering 2).
- Then lane 0 releases.
- racecheck: `review`, with only the WGMMA advisory (was `data_race`) in both checked configs.
- 20/20 tests; error-mode compile clean; SASS chains unchanged.
- Gated geomean vs DeepGEMM **0.998x / 0.997x** on two runs, against 1.001x before: about 0.3–0.4%.
- **Accepted** (review, 2026-11-21): 0.3% is the right price for a fence the model requires.
- **Follow-up for tvm-tirx (open):** expose the uni-directional, read-only release fence
  `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster`, the one the guide's example uses,
  in the PTX table. It orders only prior generic *reads* before later async-proxy accesses, which is exactly
  this WAR, and may recover part of the 0.3%. Re-measure B2 with it once it exists. Not filed with
  sm90-codegen-fix yet; it is an ISA-coverage item rather than a codegen bug.
- B1 has no generic read of TMA-written smem. Its WGMMA reads and TMA writes are both async proxy, and its
  epilogue already fences its `stmatrix` writes before the TMA store. It needs no change.
