"""Seed: the B3 sm_90a FA3-style attention forward from tirx-kernels (haok/sm90-fa3)."""

from tirx_kernels.attention import fa3_forward_sm90 as fa

BC = {"cuda": {"arch": "sm_90a"}}


def setup(data, seq_lens, num_heads, causal):
    launcher = fa.Launcher(num_heads, causal, backend_config=BC)
    run = launcher.bind(data["Q"], data["K"], data["V"], data["O"], seq_lens)
    run()
    return run
