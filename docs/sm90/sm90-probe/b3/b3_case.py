"""One fa3_forward_sm90 case: argv lens(comma) heads causal(0/1)."""
import sys, torch
from tirx_kernels.attention import fa3_forward_sm90 as fa
lens = tuple(int(x) for x in sys.argv[1].split(",")); H = int(sys.argv[2]); causal = sys.argv[3] == "1"
L = fa.Launcher(lens, H, causal, backend_config={"cuda": {"arch": "sm_90a"}})
q, k, v = fa.prepare_data(lens, H)
o = torch.full_like(q, float("nan"))
print("launch", lens, H, causal, flush=True)
L.bind(q, k, v, o)(); torch.cuda.synchronize()
print("kernel done", flush=True)
r = fa.reference(q, k, v, lens, causal); torch.cuda.synchronize()
print("reference done", flush=True)
print("done max err", (o.float() - fa.reference(q, k, v, lens, causal)).abs().max().item(), flush=True)
