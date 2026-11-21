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

Neither was changed yet: both need a decision (and #1 a re-measurement).
