"""The B2 output check must reject what calc_diff < 1e-3 alone accepts."""
import torch
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
M, N, K = 4096, 7168, 2048
a8, sfa, b8, sfb, d = g.prepare_data(M, N, K)
ex = g.compile_persistent(M, N, K, backend_config={"cuda": {"arch": "sm_90a"}})
ex(a8, g.sfa_mn_major(sfa), b8, sfb, d)
g.check_output(d, a8, sfa, b8, sfb)
print("clean output: pass")
ref = g.reference(a8, sfa, b8, sfb, out_dtype=torch.float64)
for name, mutate in [("row 5 := row 6", lambda x: x[5].copy_(x[6])),
                     ("row 100 zeroed", lambda x: x[100].zero_()),
                     ("one element +1 ulp x 4", lambda x: x[7, 9].copy_(x[7, 9] * (1 + 2**-5))),
                     ("last bit flipped in 1 element", lambda x: x.view(torch.int16)[3, 3].bitwise_xor_(1))]:
    bad = d.clone(); mutate(bad)
    cd = g.calc_diff(bad, ref)
    try:
        g.check_output(bad, a8, sfa, b8, sfb)
        print(f"{name}: calc_diff {cd:.2e} -> NOT caught")
    except AssertionError as e:
        print(f"{name}: calc_diff {cd:.2e} (< 1e-3: {cd < 1e-3}) -> caught: {str(e)[:110]}")
