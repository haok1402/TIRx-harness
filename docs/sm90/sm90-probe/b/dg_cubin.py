import os, sys, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import deep_gemm
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
M, N, K = 4096, 7168, 2048
a8, sfa, b8, sfb, d = g.prepare_data(M, N, K)
deep_gemm.fp8_gemm_nt((a8, deep_gemm.get_mn_major_tma_aligned_tensor(sfa)), (b8, sfb), d)
torch.cuda.synchronize()
print("ok")
