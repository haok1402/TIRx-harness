import os, re
os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
from tirx_kernels.runner import compile_kernel
BC = {"cuda": {"arch": "sm_90a"}}
for kw in (dict(_cid_from_bid=False), dict(_cid_from_bid=True)):
    ex = compile_kernel(g.make_kernel(4096, 4096, 7168, bm=256, bn=128, cluster="n", regs=(40, 232), backend_config=BC, **kw), backend_config=BC)
    src = ex.mod.imports[0].inspect_source()
    i = src.find("fetch_register_clusterid_x()")
    print(kw, "helper:", re.search(r"__device__[^{]*fetch_register_clusterid_x\(\) \{[^}]*\}", src).group(0).replace("\n", " ") if i >= 0 else "-")
    body = src[src.find("main_kernel(const"):]
    print("  uses in kernel:", body.count("clusterid_x()"), " '/' ops:", len(re.findall(r" / \d", body)), " '%' ops:", len(re.findall(r" % \d", body)))
    k = body.find("for (int kb")
    print("  first k-loop body excerpt:", body[k:k + 1400].replace("\n", " ")[:1400])
