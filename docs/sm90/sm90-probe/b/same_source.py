"""Generated CUDA for a kernel module from two TIRx-kernels trees must be byte-identical. argv: tree_a tree_b module arch hs..."""
import os, sys, importlib, hashlib
os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
import tvm_tirx
tree_a, tree_b, modname, arch, *hss = sys.argv[1:]
def src(tree, hs):
    for k in [k for k in sys.modules if k.startswith("tirx_kernels")]:
        del sys.modules[k]
    sys.path.insert(0, tree)
    try:
        m = importlib.import_module(modname)
        from tirx_kernels.runner import compile_kernel
        bc = {"cuda": {"arch": arch}}
        return compile_kernel(m.get_kernel(int(hs), backend_config=bc), backend_config=bc).mod.imports[0].inspect_source()
    finally:
        sys.path.remove(tree)
for hs in hss:
    a, b = src(tree_a, hs), src(tree_b, hs)
    print(arch, hs, "identical" if a == b else "DIFFERENT", hashlib.sha1(a.encode()).hexdigest()[:10], hashlib.sha1(b.encode()).hexdigest()[:10])
