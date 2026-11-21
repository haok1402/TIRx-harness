"""Seed: the B0 sm_90a RMSNorm from tirx-kernels (haok/sm90-rmsnorm)."""

from tirx_kernels.norm import rmsnorm as m
from tirx_kernels.runner import compile_kernel

BC = {"cuda": {"arch": "sm_90a"}}


def setup(data, batch_size, hidden_size):
    ex = compile_kernel(m.get_kernel(hidden_size, backend_config=BC), backend_config=BC)
    args = m._kernel_args(data["X"], data["W"], data["Y"])

    def run():
        ex(*args)

    run()
    return run
