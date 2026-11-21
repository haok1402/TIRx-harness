## Leg 1 (14:46-15:07)

turns 74, 21.0 min, $4.20

I didn't reach the 1.1× goal. The best passing candidate scores **1.014× overall** and **1.002× for batch_size ≥ 4113**, with all 15 shapes passing correctness. I measured it twice with the official benchmark and got the same numbers both times. From my measurements, 1.1× doesn't look reachable on this setup.

**Why 1.1× is out of reach:** about 94% of FlashInfer's summed time is the six large shapes, and those are limited by HBM bandwidth. FlashInfer already reaches about 4.06 TB/s at 16384×8192. Every variant I tried that moves the data with the GPU's SMs topped out at the same rate, including:
- a version with the reduction removed (a pure copy);
- TMA bulk loads and stores;
- L2 cache-policy hints (`createpolicy` with evict_first/evict_last);
- streaming store hints;
- higher occupancy.

Only the driver's own memcpy, which is not allowed here, reached about 4.2 TB/s, about 4% headroom. If the large shapes are pinned at roughly 1.0× and the small ones at roughly 1.3×, the overall cap is about 1.02–1.06×.

**Per-shape results for the best candidate (`frontier/hybrid-regres-tmaring`):**

| batch_size | speedup vs FlashInfer |
|---|---|
| 1, 32, 128 | 1.13–1.35× |
| 4113 | 1.07× (hs4096), 1.10× (hs5120), 1.01× (hs8192) |
| 16384 | 0.98–0.995× |

**What the hybrid does:** it picks one of two kernels per shape.
- **Small batches, all of hidden size 8192, and hs5120 at 16384 rows:** one row per CTA, held entirely in registers. X is read from memory once and the W loads overlap the reduction.
- **hs4096 large batches and hs5120 at 4113 rows:** a fixed set of CTAs per SM (4 per SM, each looping over rows) streams whole rows into a 5–6 stage shared-memory buffer with TMA bulk copies. W stays in registers for the whole run. This is what wins the 4113-row shapes, because it avoids the slow last wave.

**Frontier** (`candidates/rmsnorm_h200/frontier/index.json` is up to date and committed):
- `hybrid-regres-tmaring`: 1.014×, current best.
- `register-onepass`: 1.011×, the register-only kernel.
- `persistent-tma-ring`: 0.999×, the TMA-pipeline kernel; it loses at 8192×16384 (0.957×).

**Benchmark runs used: 4 of 20.** That was the hybrid, the register-only kernel, the TMA-pipeline kernel, and a rerun of the hybrid. I did about 10 additional diagnostic sweeps through `kcoral_remote.py --cmd`. They time candidates against FlashInfer with the same CUPTI cold-L2 timer, and I did not count them as benchmark runs.

**What blocked or misled me:**
- **Remote tools fail auth:** `kcoral_python.py`, `kcoral_ncu.py`, `kcoral_iket.py` and `kcoral_compute_sanitizer.py` don't accept `--remote @FILE`, which PROMPT.md documents. Given the plain URL, they still fail with HTTP 401, because only `kcoral_remote.py` sends the token. So I couldn't run IKET, NCU or Compute Sanitizer at all.
- **Health check doesn't work as written:** the documented `curl …/health` check also returns 401.
- **No `python` on the server's PATH:** `--cmd` scripts need the server venv's interpreter path spelled out in full.
- **Starting kernel needs a forbidden flag:** the sm_90 path in the installed `tirx_kernels/norm/rmsnorm.py` relies on `allowed_func_calls` for its packed f32x2 multiply. The task forbids that flag, so I used scalar multiplies instead.
- **Bash hook false positive:** one harmless command was blocked by the banned-path hook. It seemed to match the working directory, which sits under a banned prefix.
- **Compiler remarks:** none worth mentioning. No kernel in any frontier candidate has spills or C75xx remarks. The only spills came from forced `min_blocks_per_sm` experiments, which I dropped.

## Leg 2 (resumed session, 15:09-15:41)

turns 51, 31.8 min, $9.20

The best candidate now scores **1.053× overall and 1.041× for batch_size ≥ 4113**, up from 1.014× / 1.002×. It passed all 15 shapes in three official runs (1.051×, 1.051×, 1.053×). That is still short of the 1.1× goal, and I don't think 1.1× is reachable here.

**Benchmark runs used: 12 of 20 total** (8 of the 16 in this continuation). I stopped at about 15:41 with time left because the remaining ideas were each worth under ~1%.

**What produced the gains:**
1. **Removing `L1::no_allocate` from the X loads** (+2–3% on the large shapes). Plain `ld.global.nc.v4` is faster on H200. With this change every large shape beats FlashInfer instead of losing by 1–2%. Result: 1.039×.
2. **Keeping the rows streaming in order through memory.** Shuffling the row order costs 7–8%, and splitting into 2–16 interleaved streams costs 1–5%. The persistent kernels lost at 16384 rows because each CTA processed a fixed set of rows, the CTAs drifted apart, and the row stream scattered. I fixed this by handing out rows from a global atomic counter (fetched one row ahead; the last CTA resets it, with no fence). Result: 1.044×.
3. **Retuning those kernels.** Using fewer CTAs per SM with deeper buffers, and issuing the first loads before the CTA-wide sync. Result: 1.050–1.053×.
4. **Plain W loads for hs5120 small batches** (e.g. 1.47 vs 1.86 µs at batch 1).

**Where it stands:** at 16384×8192 the kernel moves data at about 4.15 TB/s (129.5 µs), against about 127 µs for the GPU driver's own memory copy of the same bytes. So under 2% headroom is left on the shapes that dominate the score. Small batches are at 1.13–1.49× and limited by fixed latency.

**Frontier** (`frontier/index.json` is updated and committed):

| Member | Approach | Score |
|---|---|---|
| `persistent-tma-ring` (global best) | Persistent CTAs fill a multi-stage shared-memory ring of whole rows using TMA bulk loads, taking rows from the atomic counter; the register kernel handles batch ≤ 1024 | 1.053× / 1.041× |
| `register-onepass` | One row per CTA, entirely in registers, no shared memory | 1.036× / 1.024× |

I retired the earlier hybrid because these two members together cover what it did.

**Tried and dropped:**
- Splitting each row across a cluster of CTAs for small batches.
- A persistent kernel double-buffering rows in registers instead of shared memory.
- TMA bulk stores for the output.
- L2 cache eviction hints (these cost 3–5%).
- Raising or lowering occupancy.
- A dedicated producer warp for the TMA ring (about 3% slower).

**Blockers and misleading points:**
- **Dedicated tool scripts can't connect:** `kcoral_python.py`, `kcoral_ncu.py`, `kcoral_iket.py` and `kcoral_compute_sanitizer.py` still don't accept `--remote @FILE` and fail with HTTP 401 because they send no token. So NCU, IKET and Compute Sanitizer were unavailable. I measured with my own CUPTI cold-L2 timing sweeps run through `kcoral_remote.py --cmd`, which I didn't count as benchmark runs.
- **My first conclusion was wrong:** last session I said FlashInfer was already at the bandwidth ceiling. That was only because my kernels carried the `L1::no_allocate` penalty, which is easy to assume is helpful for streaming reads.
- **The first counter-reset version was slow:** I used a full `fence.acq_rel.gpu` before the reset atomic, which cost about 1 µs at the end of each 4113-row run until I removed it.
- **Server preemption:** it happened once and the commands recovered on their own after about 1.5 minutes.
- **Compiler remarks:** local ptxas (CUDA 13.2) reports 8-byte spills in some ring configs, but the server's compiler printed no remarks for any scored candidate.

