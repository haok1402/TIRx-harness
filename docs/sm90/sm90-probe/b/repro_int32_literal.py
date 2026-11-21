"""Minimal repro (tvm-tirx upstream main 6569e2a18): tirx.LowerIntrin raises
  ValueError: Check failed: ... Literal value 6392509760 exceeds range of int32
from IntImm::IntImm via CanonicalSimplifier <- AnalyzerObj::IngestBinding <- SyncConstraintContext
(StmtExprMutatorWithConstraintContext), i.e. while ingesting a let binding whose canonical form has an
out-of-int32 constant. The indices themselves stay far inside int32 (C has 45M elements).
N = 8192 compiles; N = 11008 (= 43 * 256) fails. Each variant below drops one ingredient."""
import os
os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
import tvm_tirx
from tvm_tirx.script import tirx as T

M, N = 4096, 11008
NT = N // 256  # 43


def build(variant):
    cluster = 2 if variant == "cluster_id" else 1
    @T.function
    def main(C: T.Tensor((M, N), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=132, block=256, cluster=cluster))
        tx = T.cuda.thread_idx("x")
        bid = T.cuda.block_idx("x")
        for j in T.serial(11):
            if T.constexpr(variant == "no_div2"):
                tp = bid + j * 66
            elif T.constexpr(variant == "cluster_id"):
                tp = T.cuda.cluster_id("x") + j * 66
            else:
                tp = bid // 2 + j * 66
            if tp < 688:
                if T.constexpr(variant == "no_tx_term"):
                    row = (tp // NT) * 256
                else:
                    row = (tp // NT) * 256 + tx // 4
                C[row, 0] = T.float32(1.0)
    return main


for v in ("full", "no_div2", "no_tx_term", "cluster_id"):
    try:
        tvm_tirx.compile(tvm_tirx.IRModule({"main": build(v)}),
                         target=tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"}), pipeline="tirx")
        print(f"{v}: ok")
    except Exception as e:  # noqa: BLE001
        print(f"{v}: FAIL {str(e).splitlines()[-1][-60:]}")
