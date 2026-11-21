# B3: FA3-style attention forward on H200 (bf16, hd128, causal and non-causal)

Target: ≥ 0.90x of FA3 (flashinfer `backend="fa3"`) at every gated shape (`logs/b3/baseline.log`:
556–650 TFLOPS), C75xx-clean under `perf_remarks="error"`. Built with the A7 helpers (`SmemTile`, chains,
`NamedBarriers`, `broadcast_warpgroup_id`) on `haok/sm90-stack-fixes` (ISSUE-001..003 merged; wheel
`apache_tvm_tirx-0.1.1.dev27+g801280ac4`, venv `/mnt/weka/shrd/ci/haok/TIRx-kernels/b3env`, `b/run_b3.sh`).

## Risk retired first: does FA3's intra-warpgroup overlap compile clean? (`b3_overlap_probe.py`)

B2-design saw ptxas serialize every wgmma (C7514) when a retired accumulator was touched while another
chain was in flight. FA3 does exactly that twice per KV block (FlashInfer `hopper/mainloop_mma.cuh`):
rescale O while the S = QKᵀ chain runs, then softmax on S while the O += PV chain runs.

The probe is one consumer warpgroup, Q[64,128] against 128-row KV blocks, written in tirx-lite with
`perf_remarks="error"`, checked against torch:

| Order | Blocks | Registers | Remarks | SASS chains | Max error |
|---|---|---|---|---|---|
| serial (wait 0 after every chain) | 3 unrolled | 194 | none | 6 × 8 GMMA | 6e-4 |
| FA3 overlap | 3 unrolled in Python | > 255 | 184 B spill (no C7514) | — | — |
| **FA3 overlap, runtime loop (8 blocks)** | peeled 2 + loop | **186** | **none** | **6 × 8 GMMA** | **5e-4** |

- **No C7514 in either overlap.** Touching the retired accumulator while another chain is in flight is
  fine here. (B2's case read a retired *temporary* that was a second accumulator of the same chain shape;
  revisit if it reappears.)
- **Register budget:** 186 for S (64) + O (64) + P (32 packed bf16) + stats + descriptors. That fits the 240
  of FA3's consumer `setmaxnreg`. Fully unrolling the KV loop in Python keeps every chain's descriptors
  live and spills, so the KV loop must be a runtime loop.
- **Bugs found in writing it** (not codegen):
  - the running row sum `l` is lane-local in the loop, as in FA3, and must be quad-reduced once at the
    end;
  - the single-block final step must not rescale or accumulate into an uninitialized O.

## Kernel structure (first version)

- **Layout:** FlashInfer ragged NHD, `[b·s, h, d]`, as the baseline uses. TMA 3-D maps
  `(d, h, tokens)`, box `(64, 1, 128)`, 128B swizzle; one box per 64-column atom of hd128.
- **CTA:** 3 warpgroups, 384 threads, `setmaxnreg` 24 / 240 / 240, `min_blocks_per_sm=1`.
  - **WG0, producer:** one elected thread issues Q once per tile and K/V into a 2-stage ring (full =
    TMA-tx mbarrier, empty = consumer arrivals, per warp as in B1/B2).
  - **WG1, WG2:** consumers, 64 Q rows each of a 128-row Q tile, sharing every K/V stage.
- **Per consumer warpgroup, per KV block:** the probe's loop: S chain; rescale O; PV chain (rs, P in
  registers, V MN-major `transpose_b`); `wait 1`; release K; softmax; `wait 0`; release V; P = bf16(S).
- **Inter-warpgroup ping-pong** (FA3 `WarpScheduler`): named barriers make WG1 issue its GEMMs while
  WG2 runs softmax, and vice versa. The warpgroup index comes from `T.cuda.broadcast_warpgroup_id()`
  at the top of the consumer branch.
- **Tiles:** BM 128 (2 × 64), BN 128. FA3 uses BN 176 non-causal; try that once 128 works (S grows to
  88 registers).
- **Shared memory:** Q 32 KB + K 2 × 32 KB + V 2 × 32 KB = 160 KB; O staging reuses Q.
- **Causal:**
  - KV blocks run from the diagonal down. The first `ceil(BM / BN)` blocks are masked, the rest
    unmasked, as in FA3.
  - Q tiles are scheduled longest first, for balance.
  - Start non-persistent; add a persistent scheduler if the tail costs.
- **Epilogue:** O / l to bf16, stored to shared memory, then a TMA store.
- **Correctness:** against a float32 torch reference (SDPA math), bf16 tolerance. Also cross-checked
  against FA3's output.
- **Gate:** `b3_baseline.py`-style interleaved, L2-flushed timing against FA3 in the same process, at
  all 5 gated shapes, causal and non-causal.
