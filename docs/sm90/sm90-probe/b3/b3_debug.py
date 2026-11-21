"""Find which varlen-causal case hangs: one compiled kernel, cases launched in order, flush before each."""
import sys, torch
from tirx_kernels.attention import fa3_forward_sm90 as fa
BC = {"cuda": {"arch": "sm_90a"}}
H = 3
L = fa.Launcher((128,), H, True, backend_config=BC)
for lens in [(1,), (127,), (129,), (640,), (1777,), (1, 127), (127, 129), (129, 640), (1, 127, 129, 640, 1777)]:
    print("launch", lens, flush=True)
    L.seq_lens = lens
    L.cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    L.tiles, L.num_tiles = fa.tile_table(lens, H, True)
    q, k, v = fa.prepare_data(lens, H)
    o = torch.full_like(q, float("nan"))
    L.bind(q, k, v, o)()
    torch.cuda.synchronize()
    err = (o.float() - fa.reference(q, k, v, lens, True)).abs().max().item()
    print("done", lens, "max err", err, flush=True)
