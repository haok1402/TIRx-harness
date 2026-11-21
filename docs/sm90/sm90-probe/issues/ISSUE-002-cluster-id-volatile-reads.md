# ISSUE-002: `T.cuda.cluster_id` / `cluster_cta_id` are re-read with `asm volatile` at every use (~10% in B2)

- **Found by:** B2, `fp8_blockscale_gemm_sm90` with a 2-CTA cluster, on H200.
- **Affects:** tvm-tirx upstream main `6569e2a18` and the dev stack. This is codegen, independent of arch.
- **Repro:** `sm90-probe/b/repro_cluster_id_bounds.py` (compile only), log `logs/b2/repro_cluster_id.log`.
  Kernel A/B: `b/b2_fp8.py`, logs `logs/b2/v2_cid.log` and `logs/b2/v2_cidlocal.log`.

## What happens

`T.cuda.cluster_id("x")` lowers to a call of

```c
__device__ int32_t tvm_builtin_ptx_fetch_register_clusterid_x() {
    uint32_t x; asm volatile("mov.u32 %0, %clusterid.x;" : "=r"(x)); return (int32_t)x; }
```

`cluster_cta_id` lowers to the matching `cluster_ctaid_x()` call. The call is emitted again **at every
use** of the bound Python name. Two consequences:

- **Repeated volatile reads.** The cluster index flows into the tile coordinates, so a persistent kernel
  gets dozens of volatile asm statements, many inside its k-loops:
  - B2 kernel: 35 `clusterid` reads, plus the `cluster_ctaid` ones.
  - In the producer's k-loop, the TMA coordinates re-read both registers on every iteration.

  nvcc can neither CSE nor hoist `asm volatile`.
- **No value range.** `floordiv` and `floormod` of expressions that contain the call keep their
  negative-operand fix-up code, `+ ((x % 224) >> 31)`. With `block_idx("x")` the analyzer knows the
  range and removes it.

## Evidence

H200, same step, interleaved, ratio against DeepGEMM. The only difference between the three builds is
how `cid` and `rank` are obtained:

| Shape | `cluster_id()` per use | `block_idx // 2`, `% 2` | `cluster_id()` read once into a local |
|---|---|---|---|
| 4096x7168x2048 | 0.81–0.82 | 0.932–0.956 | 0.925 |
| 4096x4096x7168 | 0.81–0.83 | 0.886–0.915 | 0.893 |
| 4096x32768x512 | 0.89–0.90 | 0.914–0.921 | 0.916 |

The minimal kernel in the repro shows the source-level difference: 19 volatile reads against 3. It is
too small to show the slowdown by itself, and I have not pinned down which SASS effect costs the time.

## Expected

Bind each cluster special register **once** at kernel entry, like `threadIdx`/`blockIdx`. Use a
non-volatile asm read, since the value is constant for the CTA's lifetime, or nvcc's `__clusterIdx()`
builtin. Also give the analyzer its range: `[0, grid / cluster)` and `[0, cluster)`.

## Workaround (in B2, and to apply to B1)

```python
cl = T.alloc_local([2], "int32")
cl[0] = T.cuda.cluster_id("x")
cl[1] = T.cuda.cluster_cta_id("x")
cid, rank = cl[0], cl[1]
```

`block_idx // 2` would also avoid the problem, but it hits ISSUE-001 at N = 11008.
