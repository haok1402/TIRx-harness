"""ISSUE-002 repro: T.cuda.cluster_id / cluster_cta_id lower to an `asm volatile` special-register read that is
re-emitted at every use (never bound once, so never CSE'd or hoisted out of loops) and carries no value range.
The same kernel written with block_idx // 2 and block_idx % 2 reads blockIdx.x, which nvcc treats as a constant."""
import collections, os, re, subprocess, sys, tempfile
import tvm_tirx
from tvm_tirx.script import tirx as T

MT, NT, G, CL = 16, 28, 8, 66  # grouped raster over 16 x 28 tile pairs, 66 clusters of 2


def build(use_cluster_id):
    rounds = -(-MT * NT // CL)

    @T.function
    def main(O: T.Tensor((MT * 2, NT), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=CL * 2, block=128, cluster=2))
        if T.constexpr(use_cluster_id):
            cid = T.cuda.cluster_id("x")
        else:
            cid = T.cuda.block_idx("x") // 2
        rank = T.cuda.cluster_cta_id("x")
        for j in T.serial(rounds):
            t = cid + j * CL
            if t < MT * NT:
                first = t // (G * NT) * G
                gsize = T.min(MT - first, G)
                m = (first + (t % (G * NT)) % gsize) * 2 + rank
                n = (t % (G * NT)) // gsize
                for r in T.serial(4):  # stands in for a k-loop that re-uses the tile coordinates
                    if T.cuda.thread_idx("x") == r:
                        O[m, n] = t + r
    return main


target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
for use in (False, True):
    os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
    src = tvm_tirx.compile(tvm_tirx.IRModule({"main": build(use)}), target=target, pipeline="tirx").mod.imports[0].inspect_source()
    del os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"]
    body = src[src.find("main_kernel("):]
    lines = [l.strip() for l in body.splitlines() if "gsize" in l or "min(" in l or "/ " in l or "% " in l][:6]
    from tvm_tirx.backend.cuda.backend_config import compiler_options, resolve_backend_config
    from tvm_tirx.support import nvcc
    binary, _ = nvcc._compile_cuda_nvrtc(src, "cubin", "sm_90a", compiler_options(resolve_backend_config({"arch": "sm_90a"})))
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(bytes(binary)); f.flush()
        sass = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
    ops = collections.Counter(m.split(".")[0] for m in re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9.]+)", sass))
    print(f"cid = {'cluster_id(x)' if use else 'block_idx(x) // 2'}: volatile register reads in kernel source: "
          f"{body.count('fetch_register_cluster')}, SASS S2R/S2UR {ops['S2R'] + ops['S2UR']}, total SASS {sum(ops.values())}")
    for l in lines:
        print("   ", l[:160])
