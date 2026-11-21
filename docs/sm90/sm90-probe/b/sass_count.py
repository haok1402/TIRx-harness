"""SASS instruction mix of a kernel module from a TIRx-kernels tree. argv: tree module arch hs..."""
import collections, importlib, re, subprocess, sys, tempfile
tree, modname, arch, *hss = sys.argv[1:]
sys.path.insert(0, tree)
m = importlib.import_module(modname)
from tirx_kernels.runner import compile_kernel
from tvm_tirx.backend.cuda.runtime import occupancy
for hs in hss:
    bc = {"cuda": {"arch": arch}}
    ex = compile_kernel(m.get_kernel(int(hs), backend_config=bc), backend_config=bc)
    binary, _ = occupancy.kernel_binary(ex)
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(binary); f.flush()
        sass = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
    ops = collections.Counter(re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", sass))
    keys = ("IMAD.WIDE", "IMAD.WIDE.U32", "IADD3", "IADD3.X", "LEA", "LEA.HI.X", "IMAD", "LDG.E.128", "STG.E.128")
    print(arch, hs, "total", sum(ops.values()), {k: ops[k] for k in keys if ops[k]})
