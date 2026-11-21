# ISSUE-008: `T.cuda.broadcast_warpgroup_id()` is re-emitted at every use, so a use in a short-circuit runs a divergent full-mask `__shfl_sync` (hang)

- **Found by:** B3 (`fa3_forward_sm90`) on `haok/sm90-stack-fixes` (801280ac4), which carries the
  ISSUE-003 fix.
- **Symptom:** the causal kernel hangs on H200 for every `seq_len % 8 != 0` (2, 63, 65, 126, 127, 129,
  191, 255…). Multiples of 8 (32, 96, 120, 128, 256) and non-causal pass.
- **Severity:** silent hang, i.e. undefined behaviour, from an innocent-looking use of the builtin that
  ISSUE-003 recommends.

## What happens

The kernel computed `cw = T.cuda.broadcast_warpgroup_id() - 1` once, in Python, and used it in the causal
mask of the first KV block:

```python
q_row = m_blk * 128 + cw * 64 + row            # row, col from acc_coord(tid, r)
masked = T.Or(k_col >= seq_len, k_col > q_row)
s_acc[r] = T.Select(masked, -inf, s_acc[r])
```

`broadcast_warpgroup_id()` lowers to an expression, `__shfl_sync(0xffffffff, threadIdx.x >> 7, 0, 32)`.
TIRx expressions are trees, so it is inlined, unbound, wherever the Python value is used. In the
generated CUDA it lands in the right operand of a short-circuit `||`:

```c
v_ptr_7[0] = (((seq_len + 128) <= (n_blocks * 128 + (tid & 127 & 3) * 2))
              || (((m_blk * 64 + (__shfl_sync((uint)4294967295, (int)threadIdx.x >> 7, 0, 32) * 32) + ...) < -32))
             ? -CUDART_INF_F : v_ptr_7[0];
```

Only the lanes whose left operand is false execute the shuffle. With a full mask, that is undefined
behaviour, and on H200 the warp hangs. The left operand (the KV-tail test) differs across a warp
exactly when `seq_len % 8 != 0`: each 8-row group of the 64×N accumulator layout then straddles the
tail.

## Repro

- Build `tirx_kernels/attention/fa3_forward_sm90.py` from TIRx-kernels `haok/sm90-fa3` with
  `cw = T.cuda.broadcast_warpgroup_id() - 1` used directly, causal, one sequence of length 127.
- Or, minimal: any kernel where `x = T.cuda.broadcast_warpgroup_id()` appears in the second operand of
  `T.Or` / `T.And` / `T.if_then_else` / `T.Select` with a lane-varying first operand.
- Logs: sm90-probe `logs/b3/v1_cases3.log` (the length table) and `logs/b3/causal.cu` (generated
  source).

## Workaround (shipped in B3)

Bind it once into a local at kernel entry, where the warp is converged:

```python
wg = txl.local_scalar("int32")
txl.assign(wg, T.cuda.broadcast_warpgroup_id())
```

## Suggested fixes (for sm90-codegen-fix)

- (a) Make the builtin bind its result to a fresh local at the call site (a `LetStmt` / `assign` to a
  scalar). Uses then read a register, and no use can move the shuffle into conditional code. That is
  what "emitted at the call site" in ISSUE-003's resolution should mean.
- (b) Or: CUDA codegen must never place a `__shfl_sync` (or any `.sync`/`.aligned` intrinsic) inside a
  short-circuit operand or ternary arm. Hoist such calls out before emitting `||`, `&&` and `?:`.
- (c) Document on the builtin that its value must not be used before it is bound, until (a) lands.

The same hazard applies to any warp-collective intrinsic whose value TIRx keeps as an expression
(`T.cuda.__shfl_*`, `match.any`, `vote`…) once it is used inside `T.Or`/`T.And`/`T.Select`.
