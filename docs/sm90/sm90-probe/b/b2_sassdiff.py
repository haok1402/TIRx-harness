import collections, re, subprocess, tempfile
from tvm_tirx.backend.cuda.runtime import occupancy
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
from tirx_kernels.runner import compile_kernel
BC = {"cuda": {"arch": "sm_90a"}}
res = {}
for name, kw in [("nocluster", dict(cluster=None)), ("clusterL", dict(cluster="n", _multicast=False)), ("clusterMC", dict(cluster="n"))]:
    ex = compile_kernel(g.make_kernel(4096, 4096, 7168, bm=256, bn=128, regs=(40, 232), backend_config=BC, **kw), backend_config=BC)
    binary, _ = occupancy.kernel_binary(ex)
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(binary); f.flush()
        sass = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
        open(f"/mnt/weka/home/hao.kang/kernel-evolve/sm90-probe/logs/b2/{name}.sass", "w").write(sass)
    ops = collections.Counter(m.split(".")[0] for m in re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9.]+)", sass))
    res[name] = ops
    print(name, "total", sum(ops.values()))
keys = sorted(set().union(*res.values()), key=lambda k: -res["clusterL"][k])
for k in keys:
    a, b, c = (res[n][k] for n in ("nocluster", "clusterL", "clusterMC"))
    if a != b or b != c:
        print(f"{k:10s} {a:5d} {b:5d} {c:5d}")
