import sys, json
sys.path.insert(0, sys.argv[1] + "/tests/python/backend/cuda")
import test_pipeline_multicast as t
from tvm_tirx.backend.cuda.runtime import occupancy
out = {}
for block, kb in [(384, 192), (256, 96), (128, 32)]:
    out[f"{block}thr_{kb}KB"] = {c: occupancy.max_active_clusters(t._compile(t._persistent_like(block, kb * 256, c)), block=block, cluster=c) for c in (1, 2, 4, 8)}
print(json.dumps(out))
