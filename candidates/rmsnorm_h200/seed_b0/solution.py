"""Seed: the sm_90a RMSNorm from tirx-kernels (haok/sm90-rmsnorm: the M3-evolved kernels)."""

from tirx_kernels.norm import rmsnorm as m
from tirx_kernels.runner import compile_kernel

BC = {"cuda": {"arch": "sm_90a"}}


def setup(data, batch_size, hidden_size):
    kernel = m.get_kernel(hidden_size, batch_size, backend_config=BC)
    ex = compile_kernel(kernel, backend_config=BC)
    args = m._kernel_args(data["X"], data["W"], data["Y"])  # includes the zeroed row-ticket counter

    def run():
        ex(*args)

    run()
    return run
