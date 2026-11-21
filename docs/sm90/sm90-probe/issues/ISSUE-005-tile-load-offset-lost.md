# ISSUE-005: `test_tile_load_infers_full_global_layout_with_offset` returns wrong values (offset lost)

- **Found by:** the compiler-fix agent's full-suite run on H200.
- **Affects:** tvm-tirx upstream main `6569e2a18` (and the dev stack), on sm_90a.
  Not yet checked on sm_100.
- **Severity:** silent wrong code in tile load/store, so it may affect real kernels.

## Evidence

`tests/python/tirx/test_tile_namespace.py::test_tile_load_infers_full_global_layout_with_offset`.
The source layout is `T.TileLayout(T.S[(8, 16)] + 7)`, and the kernel loads `sub[2:4, 3:7]`. The output
equals `data[0:...]` rows 0 and 1, not rows `2:4`, columns `3:7`, offset 7:

```
ACTUAL  [[ 0, 1, 2, 3], [16, 17, 18, 19]]
DESIRED [[42, 43, 44, 45], [58, 59, 60, 61]]
```

The whole base offset of 2*16 + 3 + 7 = 42 is dropped. Root cause pending.
