# ISSUE-001: `tirx.LowerIntrin` builds an out-of-range int32 literal ("Literal value 6392509696 exceeds range of int32")

- Found by: B1 (sm_90a bf16 GEMM) at M x N x K = 4096 x 11008 x 4096, both dtypes; compile fails.
- Affects: tvm-tirx upstream main `6569e2a18` (also the dev stack); arch-independent (pure lowering).
- Repro: `sm90-probe/b/repro_int32_literal.py` (no GPU needed; FORCE_FALLBACK), log `logs/b1/repro_int32.log`.

```python
M, N = 4096, 11008; NT = N // 256  # 43
@T.function
def main(C: T.Tensor((M, N), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=132, block=256))
    bid = T.cuda.block_idx("x")
    for j in T.serial(11):
        tp = bid // 2 + j * 66
        if tp < 688:
            C[(tp // NT) * 256, 0] = T.float32(1.0)
```

| Variant | Result |
|---|---|
| as above | `ValueError: ... Literal value 6392509696 exceeds range of int32` |
| `tp = bid + j * 66` (no `// 2`) | ok |
| `tp = T.cuda.cluster_id("x") + j * 66` (cluster launch) | ok |
| N = 8192 (NT = 32) | ok |

Stack (bottom): `IntImm::IntImm` <- `CanonicalSimplifier::operator()` <- `AnalyzerObj::IngestBinding` <-
`AnalyzerObj::SyncConstraintContext` <- `StmtExprMutatorWithConstraintContext::Mutate_(FloorMod/FloorDiv/Mul/Add)`
in pass `tirx.LowerIntrin` (found with a PassInstrument). The real index values are < 45M; the literal is a
symbolic intermediate (e.g. a bound or a divisor product such as 43 x 2 x 66 x ...) that the canonical simplifier
materializes as an int32 IntImm. Expected: compute such constants in int64 / BigInt, or fall back, instead of
failing compilation.

Workaround used in B1 (documented in the kernel): derive the cluster index from `T.cuda.cluster_id("x")`
instead of `block_idx("x") // 2`.
