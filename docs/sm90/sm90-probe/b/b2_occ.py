from tvm_tirx.backend.cuda.runtime import occupancy
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
from tirx_kernels.runner import compile_kernel
BC = {"cuda": {"arch": "sm_90a"}}
for cfg in [(256, 128, "n"), (256, 128, "m"), (128, 192, "m"), (256, 128, None)]:
    ex = compile_kernel(g.make_kernel(4096, 7168, 2048, bm=cfg[0], bn=cfg[1], cluster=cfg[2], backend_config=BC), backend_config=BC)
    cs = 1 if cfg[2] is None else 2
    print(cfg, "stages", g.num_stages(cfg[0], cfg[1]), "max_active_clusters", occupancy.max_active_clusters(ex, block=384, cluster=cs))
