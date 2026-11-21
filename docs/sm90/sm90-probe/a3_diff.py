"""A3 registry diff: upstream main vs A3, per (kernel, config, arch).

At a kernel's own runtime archs (sm_100+, which VerifyPTXTarget does not check) every row must be
byte-identical. At sm_90a (checked, EXTRA_ARCHS), a row main generates source for may be rejected by
A3 only with PTXTargetError; whether main's real NVRTC also rejects it is a3_confirm.py's job.
"""

import collections
import json
import sys

LOGS = "/mnt/weka/home/hao.kang/kernel-evolve/sm90-probe/logs/a3"


def load(path):
    return {(r["kernel"], r["config"], r["arch"]): r for r in map(json.loads, open(path))}


main = load(f"{LOGS}/registry_main.jsonl")
rerun = load(f"{LOGS}/registry_main_irmod.jsonl")
rerun_kernels = {k[0] for k in rerun}
main = {k: v for k, v in main.items() if k[0] not in rerun_kernels} | rerun
a3 = load(f"{LOGS}/registry_a3.jsonl")

assert set(main) == set(a3), (len(set(main) - set(a3)), len(set(a3) - set(main)))
outcome = collections.Counter()
bad = []
for key in sorted(main):
    m, a = main[key], a3[key]
    checked = key[2] == "sm_90a"
    if m["status"] == "ok" and a["status"] == "ok":
        same = m["sha"] == a["sha"]
        outcome["identical" if same else "SOURCE DIFFERS"] += 1
        if not same:
            bad.append((key, "source differs"))
    elif m["status"] == "ok" and a["status"] == "PTXTargetError":
        outcome["rejected by A3 (sm_90a)" if checked else "REJECTED AT AN UNCHECKED ARCH"] += 1
        if not checked:
            bad.append((key, a.get("msg")))
    elif m["status"] == a["status"]:
        outcome[f"same harness/main failure: {m['status']}"] += 1
    else:
        outcome[f"STATUS DIFFERS: {m['status']} -> {a['status']}"] += 1
        bad.append((key, m["status"], a["status"], a.get("msg")))

print(f"rows {len(main)} kernels {len({k[0] for k in main})}")
for name, count in outcome.most_common():
    print(f"  {count:6d}  {name}")
by_arch = collections.Counter((k[2], main[k]["status"] == "ok") for k in main)
print("covered (main ok) by arch:", sorted(by_arch.items()))
rejected = collections.Counter(k[0] for k in a3 if a3[k]["status"] == "PTXTargetError")
print("A3-rejected kernels at sm_90a:", len(rejected))
print("PROBLEMS:", len(bad))
for item in bad[:20]:
    print("  ", item)
sys.exit(1 if bad else 0)
