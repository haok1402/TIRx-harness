"""Seed: the B1 sm_90a GEMM from tirx-kernels (haok/sm90-gemm)."""

from tirx_kernels.gemm import fp16_bf16_gemm_sm90 as g

BC = {"cuda": {"arch": "sm_90a"}}


def setup(data, M, N, K):
    ex, _ = g.compile_persistent("fp16", M, N, K, backend_config=BC)
    A, B, D = data["A"], data["B"], data["D"]

    def run():
        ex(A, B, D)

    run()
    return run
