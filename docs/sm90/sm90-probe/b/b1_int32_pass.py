import tvm_tirx
from tvm_tirx.ir.transform import PassContext, pass_instrument
from tirx_kernels.gemm import fp16_bf16_gemm_sm90 as g
from tirx_kernels.runner import compile_kernel
BC = {"cuda": {"arch": "sm_90a"}}

@pass_instrument
class Rec:
    def __init__(self):
        self.names = []
    def run_before_pass(self, mod, info):
        self.names.append(info.name)
    def run_after_pass(self, mod, info):
        self.last_ok = info.name

rec = Rec()
func = g.make_kernel("bf16", 4096, 11008, 4096, backend_config=BC)
try:
    with PassContext(instruments=[rec]):
        compile_kernel(func, backend_config=BC)
except Exception as e:
    print("failed in pass:", rec.names[-1], "| previous:", rec.names[-4:-1])
