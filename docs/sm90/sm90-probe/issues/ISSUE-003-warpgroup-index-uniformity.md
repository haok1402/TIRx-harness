# ISSUE-003: a warpgroup index from `thread_idx // 128` is non-uniform, so WGMMA descriptors go through R2UR (−2% in B2)

- **Found by:** B2, `fp8_blockscale_gemm_sm90`, by diffing its SASS against DeepGEMM's `sm90_fp8_gemm_1d2d` cubin.
- **Affects:** every hand-written SM90 kernel that derives a per-warpgroup smem offset from `T.cuda.thread_idx`.
  That includes A7's own docs and tests, B1 and B2 as first written, and probably most kernels people will write.
  It is a codegen and helper-design issue, independent of arch.
- **Evidence:**
  - `sm90-probe/logs/b2/sass_vs_dg.log` (per-chain SASS mix);
  - `logs/b2/v4_fixes.log` (ablation);
  - `logs/b2/wg_ab.log` (A/B including the builtin).

## What happens

WGMMA takes its smem descriptors from uniform registers. A kernel that writes

```python
wg = T.cuda.thread_idx("x") // 128 - 1
da0 = W.smem_descriptor(A_s.ptr_to([0, wg * 64, 0]))
```

has a per-thread value as far as nvcc can tell. The 64-bit descriptor arithmetic for every stage and k-step then
runs in vector registers, and each wgmma needs an `R2UR` to move its descriptor into a uniform register.

| B2 SASS, one chain of 4 QGMMA + 64 FFMA | Non-FFMA ops between ARRIVE and the 4th QGMMA | R2UR |
|---|---|---|
| `wg` from `thread_idx` | ~40, among them `USHF.R.U64`, `LEA.HI.X`, `IMAD`, `LOP3` | 12 |
| DeepGEMM (`math_wg_idx = __shfl_sync(…, threadIdx.x / 128, 0)`) | 2 (FMUL) | 0 |

## Cost (H200, same step, gated geomean against DeepGEMM)

| B2 warpgroup index | Geomean |
|---|---|
| `thread_idx // 128` | 0.980 |
| explicit `shfl.sync.idx` broadcast from lane 0 (what B2 ships) | 0.998 |
| `T.cuda.warpgroup_id()` | 0.992 |

B1 (bf16, m64n256k16 chains) does not move with any variant (0.988–0.989). Its chains are long enough that
descriptor work is hidden.

## Notes for the fix

- `T.cuda.warpgroup_id()` exists and is already uniform by construction. `exec_analysis.cc` binds
  `warp_id_in_cta` once, through `gpu_warp_shuffle`. But nothing points kernel authors to it:
  - the A7 helpers, their docstrings and tests all use `thread_idx // 128`;
  - `wgmma.smem_descriptor` accepts any pointer silently.

  Measured, it is slightly behind the explicit broadcast (0.992 vs 0.998), probably because it shuffles the
  *warp* id and then divides by 4, instead of broadcasting the warpgroup id itself. That is worth a look.
- **Suggested fixes:**
  - (a) Document `T.cuda.warpgroup_id()` in the A7 helpers (`wgmma.py` module docs, `SmemTile`) and use it in
    their tests and examples.
  - (b) Have `warpgroup_id` broadcast `tid // 128` directly. Either way, consider a uniformity hint
    (`__builtin_assume`-style, or a `.uniform`-friendly lowering) for `warpgroup_id`/`warp_id`.
  - (c) Optionally, an A10-style remark when a wgmma descriptor operand is computed from a non-uniform
    value. The PTX → SASS R2UR count per wgmma is a cheap signal: `sass_chain_check.py` could report it.
- **Workaround (B2):**

  ```python
  wgv = T.alloc_local([1], "uint32")
  T.ptx.shfl_sync.idx.b32(wgv[0], T.Cast("uint32", tx // 128), T.uint32(0), T.uint32(31), T.uint32(0xFFFFFFFF))
  wg = T.Cast("int32", wgv[0]) - 1
  ```
