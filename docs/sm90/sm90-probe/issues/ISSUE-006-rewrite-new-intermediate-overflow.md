# ISSUE-006: rewrites that introduce an intermediate the original never computed, and the int32 index headroom

- **Found by:** the ISSUE-004 differential fuzz (compiler-fix agent); the review asked for it as a follow-up.
- **Affects:** tvm-tirx upstream main, `src/sym/rewrite_simplify.cc`, and the int64 → int32 index narrowing.
- **Severity:** wrong value only near the int32 limits. It relies on TVM's no-overflow assumption.
- **Priority:** lower, after ISSUE-003.

## What happens

After ISSUE-004, 0–2 per ~3600 fuzz expressions still evaluate wrong in `rewrite_simplify` / `simplify` (0 in
`canonical_simplify`). All of them are reassociations that create a non-constant intermediate the original
expression never computed:

| input | rewritten | wrong at |
|---|---|---|
| `T.min(536870912, x) - x` | `T.min(536870912 - x, 0)` | x = -2043769537 |
| `T.min(0 - (T.max(x, -536870911) - x), c)` | `T.min(T.min(0, x + 536870911), c)` | x = 1902339669 |
| `(T.min(185, x) + (1073741824 + x)) % 1048575` | `(T.min(185, x) + x + 1024) % 1048575` | x = -1279865054 |

Rule families involved:
- `min(x, y) - y -> min(x - y, 0)` and its max/min variants (`rewrite_simplify.cc` around 442–470);
- `(x + c) + y -> (x + y) + c`;
- `min(c1 - x, c2) -> c1 - max(x, c1 - c2)`.

## Plan

1. Bounds-guard the rules that introduce such a term, as ISSUE-004 did for the offset normalizations: rewrite
   only if the analyzer's bound on the new term fits the type. Measure how much simplification this loses on
   the tirx and kernel suites. These rules are core, so a guard may need to be scoped, e.g. applied only when
   the operands' bounds are not already proven small.
2. **int32 headroom.** `NarrowDataType(32)` (default pipeline, `compilation_pipeline.py`) narrows an int64
   index to int32 when its constant bound fits [-2^31, 2^31 - 1]. Review suggests a threshold of 2^30
   elements. Then the intermediates these rules introduce (at most about twice an index value) still fit.
   Also check:
   - where shapes and loop extents default to int32;
   - `storage_rewrite.cc`, which switches an allocation to int64 above 2^31 - 1 elements;
   - `force_narrow_index_to_i32`.
3. Extend `tests/python/sym/test_simplify_differential.py` to more seeds, and to bounded variables near 2^30.
