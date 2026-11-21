"""Per-k-block SASS mix: TIRx fp8_blockscale_gemm_sm90 vs DeepGEMM sm90_fp8_gemm_1d2d (same config 256x128x128, 3 stages).
Splits the math warpgroups' code at WARPGROUP.ARRIVE (one per wave chain) and reports the median segment's mix."""
import collections, glob, re, subprocess, sys, tempfile
from tvm_tirx.backend.cuda.runtime import occupancy
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
from tirx_kernels.runner import compile_kernel

def sass_of(path):
    return subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True).stdout

def segments(sass):
    ins = re.findall(r"/\*[0-9a-f]{4}\*/\s+((?:@!?U?P\w+\s+)?[A-Z][A-Z0-9_.]+[^;]*);", sass)
    segs, cur = [], None
    for i in ins:
        op = i.split()[1] if i.startswith("@") else i.split()[0]
        if op.startswith("WARPGROUP.ARRIVE"):
            if cur is not None:
                segs.append(cur)
            cur = []
        if cur is not None:
            cur.append(op)
    return segs, ins

def mix(seg):
    c = collections.Counter(o.split(".")[0] for o in seg)
    return c

out = {}
BC = {"cuda": {"arch": "sm_90a"}}
ex = compile_kernel(g.make_kernel(4096, 7168, 2048, bm=256, bn=128, cluster=None, regs=(40, 232), backend_config=BC), backend_config=BC)
binary, _ = occupancy.kernel_binary(ex)
with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
    f.write(binary)
    ours = f.name
dg = glob.glob(sys.argv[1] + "/cache/kernel.sm90_fp8_gemm_1d2d.*/kernel.cubin")[0]
for name, path in (("tirx", ours), ("deepgemm", dg)):
    sass = sass_of(path)
    open(f"/mnt/weka/home/hao.kang/kernel-evolve/sm90-probe/logs/b2/{name}_256x128.sass", "w").write(sass)
    segs, ins = segments(sass)
    gm = [s for s in segs if any(o.startswith(("HGMMA", "QGMMA")) for o in s)]
    lens = sorted(len(s) for s in gm)
    print(f"== {name}: total SASS {len(ins)}, chain segments {len(gm)}, segment lengths {lens}")
    tot = collections.Counter()
    for s in gm:
        tot += mix(s)
    n = len(gm)
    print("  per-segment average:", {k: round(v / n, 1) for k, v in tot.most_common(28)})
    # show one representative steady-state segment
    med = sorted(gm, key=len)[len(gm) // 2]
    print("  median segment ops:", " ".join(med)[:3000])
