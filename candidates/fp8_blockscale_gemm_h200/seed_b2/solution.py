"""Seed: the B2 sm_90a FP8 block-scaled GEMM from tirx-kernels (haok/sm90-fp8-gemm)."""

from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g

BC = {"cuda": {"arch": "sm_90a"}}


def setup(data, M, N, K):
    ex = g.compile_persistent(M, N, K, backend_config=BC)
    args = (data["A"], data["SFA"], data["B"], data["SFB"], data["D"])

    def run():
        ex(*args)

    run()
    return run
