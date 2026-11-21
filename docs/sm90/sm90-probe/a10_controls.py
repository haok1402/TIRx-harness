# SPDX-License-Identifier: Apache-2.0
"""Run hand-written / previously generated .cu files through compile_source (A10) and print remarks."""
import glob, json, os, sys, warnings
from tvm_tirx.backend.cuda.compiler import compile_source
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.cu"))):
    src = open(path).read()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = compile_source(src, {"cuda": {"arch": "sm_90a"}})
            rec = {"file": os.path.basename(path), "remarks": [str(x) for x in r.remarks]}
        except Exception as e:  # noqa: BLE001
            rec = {"file": os.path.basename(path), "error": f"{type(e).__name__}: {e}"[:300]}
    rec["warnings"] = sum(x.category.__name__ == "TIRxPerfWarning" for x in w)
    print(json.dumps(rec), flush=True)
