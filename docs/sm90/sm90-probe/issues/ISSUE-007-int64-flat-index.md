# ISSUE-007: a flat tensor index never switches to int64, so tensors past 2^31 elements overflow

- **Found by:** the compiler-fix agent, while working on ISSUE-006.
- **Affects:** tvm-tirx upstream main. Arch-independent.
- **Severity:** silent wrong code for tensors of at least 2^31 elements with int32 dimensions.

## What happens

The flat index of a tensor takes the dtype of its shape and its indices. Python ints in
`T.Tensor((M, N))` are int32, so

```python
A: T.Tensor((65536, 65536), "float32")   # 2^32 elements
A[bx, j * 256 + tx] = ...
# CUDA: A_ptr[(((int)blockIdx.x) * 65536) + (j * 256)) + ((int)threadIdx.x)]   <- int32, overflows
```

Under ISSUE-006's contract, int32 index values stay within |v| < 2^30. A tensor whose static span is at least
2^30 elements must therefore index in int64.

## Acceptance (from review)

- Tensors with static span >= 2^30 index in int64 and compute correctly, including a GPU test above 2^31
  elements (H200 has 141 GB).
- Generated CUDA is byte-identical for all TIRx-kernels kernels and for B1/B2 whose spans are below the
  threshold, so there is no performance change.
- For dynamic shapes, state what is done and why.
