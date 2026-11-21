# ISSUE-004: rewrite rules build overflowing constant products, so `Simplify` emits wrong code

- **Found by:** the compiler-fix agent, while fixing ISSUE-001.
- **Affects:** tvm-tirx upstream main `6569e2a18`, `src/sym/rewrite_simplify.cc`. Arch-independent.
- **Severity:** silent wrong code. The bad form is only an error when the canonical simplifier
  normalizes it (ISSUE-001).

## What happens

Many rewrite rules compute a constant from matched constants in their right-hand side, e.g.

```
floordiv(floordiv(x, c1), c2)   -> floordiv(x, c1 * c2)
floordiv(x, c1) < c2            -> x < c1 * c2
(x + c1) * c2                   -> x * c2 + c1 * c2
```

Since #26 ("Fold constants only when results fit their types"), `c1 * c2` is not folded when the
product does not fit the type, and the pattern `Eval` silently builds `Mul(c1, c2)`. That node wraps
at runtime:

```python
x = tirx.Var("x", "int32")
Analyzer().rewrite_simplify(x // 65536 // 65536)   # x // T.Mul(65536, 65536) -> divide by 0
```

About 100 rules compute constants this way. That includes the div/mod chains, the comparison rules
(`x // c1 < c2`, `c1 < x - c2`, `floordiv(x, c2) == c1 && ...`) and the min/max and add/sub folding.

## Expected

A rule whose right-hand side needs a constant that does not fit the type does not fire.
