#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Structural WGMMA chain check on SASS (cuobjdump -sass output, or a .cubin to disassemble).

A SASS chain is the GMMA instructions between one WARPGROUP.ARRIVE and the next. A healthy chain of
k MMAs has ONE ARRIVE and `gsb0` only on its LAST GMMA.

SASS alone cannot tell a ptxas-serialized chain from a kernel that deliberately commits each MMA as
its own group: both are ARRIVE + gsb0 per GMMA. **ptxas's (C75xx) remark is the authoritative signal**
(sass_check.sh's perf= column, TIRx's perf_remarks). This script adds structure:

  gsb0_not_last : a chain has gsb0 on a GMMA that is not its last                     (error)
  serialized    : with --ptx, a SASS chain is shorter than the PTX commit group it came
                  from (the PTX's wgmma.mma_async count per commit_group)               (error)
  short_chain   : a chain is shorter than --min-chain                                 (error)
  unchained     : every chain has one GMMA (serialized OR one MMA per group by design) (info)
Exit code 1 on any error flag. One JSON line per function with GMMA.
"""
import argparse, json, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("path", help=".sass text or .cubin")
ap.add_argument("--ptx", help="the PTX the cubin was built from: compare chain lengths with its commit groups")
ap.add_argument("--min-chain", type=int, default=0, help="minimum GMMAs per chain")
a = ap.parse_args()
text = (subprocess.run(["cuobjdump", "-sass", a.path], capture_output=True, text=True, check=True).stdout
        if a.path.endswith(".cubin") else open(a.path).read())

funcs, cur = {}, None
for line in text.splitlines():
    m = re.search(r"Function : (\S+)", line)
    if m:
        cur = funcs.setdefault(m.group(1), [])
        continue
    if cur is None:
        continue
    if "WARPGROUP.ARRIVE" in line:
        cur.append([])
    elif re.search(r"\b[HQIB]GMMA\.", line):
        if not cur:
            cur.append([])
        cur[-1].append("gsb0" in line)

ptx_groups = {}  # function -> list of mma_async counts per commit_group, in program order
if a.ptx:
    fn, count = None, 0
    for line in open(a.ptx):
        m = re.search(r"\.entry\s+(\w+)", line)
        if m:
            fn, count = m.group(1), 0
            ptx_groups[fn] = []
        elif fn and "wgmma.mma_async" in line:
            count += 1
        elif fn and "wgmma.commit_group" in line:
            ptx_groups[fn].append(count)
            count = 0

bad = False
for name, chains in funcs.items():
    chains = [c for c in chains if c]
    n = sum(len(c) for c in chains)
    if not n:
        continue
    flags = []
    if any(any(c[:-1]) for c in chains):
        flags.append("gsb0_not_last")
    groups = [g for g in ptx_groups.get(name, []) if g]
    if groups and len(chains) > len(groups):
        flags.append("serialized")
    if a.min_chain and any(len(c) < a.min_chain for c in chains):
        flags.append("short_chain")
    if n > 1 and all(len(c) == 1 for c in chains):
        flags.append("unchained")
    bad |= any(f != "unchained" for f in flags)
    rec = {"function": name, "gmma": n, "chains": [len(c) for c in chains],
           "gsb0": sum(sum(c) for c in chains), "flags": flags}
    if groups:
        rec["ptx_groups"] = groups
    print(json.dumps(rec))
sys.exit(1 if bad else 0)
