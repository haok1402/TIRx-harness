# ISSUE-009: `T.max` on floats lowers to a 3-instruction NaN-propagating select, and softmax's `exp2((s - m) * c)` stays FADD + FMUL (B3: 20% of the kernel's instructions)

- **Found by:** B3 (`fa3_forward_sm90`, TIRx-kernels `haok/sm90-fa3`) on `haok/sm90-stack-fixes`
  (801280ac4), by reading its SASS mix because NCU has no counter access on this cluster.
- **Affects:** every softmax-style hot loop written with `T.max` and `T.exp2` on f32 (attention, MoE
  routing, sampling…), on any arch.

## What happens

1. **`T.max(a, b)` on f32** generates CUDA whose SASS is `FSETP.GT.FTZ.AND` + `FSETP.NAN.FTZ.AND` +
   `FSEL`, three instructions. It propagates NaN, consistent with how TIRx defines `max`.
   `fmaxf` / PTX `max.f32` is a single `FMNMX`, which returns the non-NaN operand. The FA3 row max
   runs over 64 accumulator registers per thread per KV block, so that is about 128 extra instructions
   per block.
2. **`exp2((s - m) * c)`**, with `c = log2(e)/sqrt(d)`, compiles to FADD + FMUL before `MUFU.EX2`. FA3
   writes `exp2(s * c - m*c)` with `m*c` hoisted, which is one FFMA. That form is a reassociation, not
   a contraction of the expression as written: `(s - m) * c` and `s * c - m * c` round differently. So
   no compiler may rewrite one into the other without an explicit fast-math permission. About 64 more
   instructions per block.

## Measured (B3 commit `6c4c04c`, both changes together)

The changes: `T.max` replaced by `txl.ptx["max.f32"]`, and the exponent written as
`s * c - mc` with `mc = m * c`.

| | before | after |
|---|---|---|
| SASS instructions (whole kernel) | 2880 | **2304 (−20%)** |
| FMNMX | 0 (FSETP/FSETP/FSEL: 134 + 134 + 198) | 136 |
| FFMA / FMUL / FADD | 2 / 322 / 260 | 130 / 198 / 132 |
| non-causal vs FA3 (H200, same step) | 0.83–0.90x | 1.01–1.05x |

The two changes were not timed separately. Logs: sm90-probe `logs/b3/v2_pp.sass`, `v3_pp.sass`,
`v2_ab.log`, `v3.log`.

## Suggested fix

Don't change `T.max`'s semantics silently: its NaN propagation is the documented behaviour and some
code depends on it. Instead:

- (a) **An explicit NaN-ignoring float max**: e.g. `T.fmax` / `T.max(..., nan="ignore")`, or a `T.fast`
  flavour lowering to `fmaxf` / `max.f32` (`FMNMX`). Possibly also `max.NaN`-free intrinsics for f16x2
  and bf16x2. Document next to `T.max` that the default costs three instructions on f32.
- (b) **FMA contraction where allowed**: make sure `a * b + c` and `a * b - c` written by kernels reach
  nvcc in a contractible form. nvcc contracts by default (`--fmad=true`); check that TIRx neither
  disables that nor splits the expression across statements so that contraction is lost.
- (c) **An opt-in fast-math / reassociation mode**, per function or per expression (e.g.
  `T.exp2(..., fast=True)` or a `backend_config` key). It would let the simplifier rewrite
  `(s - m) * c` into `s * c - (m * c)`, hoisting the loop-invariant product, and map `exp2` to
  `ex2.approx.ftz`. Off by default.
- (d) **Optionally, an A10-style perf remark** when a hot loop contains `FSETP.NAN` selects from
  `T.max`. That is cheap to detect in the SASS mix that `sass_chain_check.py` already parses.

## Workaround (in B3)

```python
txl.ptx["max.f32"](dst, a, b)                          # one FMNMX
mc = m * c                                             # once per row
p = T.exp2(s * T.float32(c) - mc)                      # one FFMA into MUFU.EX2
```

The SM90 wiki (`skills/tirx-wiki/optimization/sm90.md` §8) documents both for kernel authors.
