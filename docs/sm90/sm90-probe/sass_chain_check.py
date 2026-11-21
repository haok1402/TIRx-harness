#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Structural WGMMA chain check on SASS (cuobjdump -sass output, or a .cubin to disassemble).

A chain is the GMMA instructions between one WARPGROUP.ARRIVE and the next. A healthy chain of k
MMAs has ONE ARRIVE and `gsb0` only on its LAST GMMA. ptxas serialization shows as one ARRIVE +
gsb0 per GMMA (chains of length 1).

Flags per function:
  gsb0_not_last : a chain has gsb0 on a GMMA that is not its last
  serialized    : the function has > 1 GMMA and every chain has length 1
  short_chain   : a chain is shorter than --min-chain (when given)
Exit code 1 if any function is flagged. Prints one JSON line per function with GMMA.
"""
import argparse, json, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("path", help=".sass text or .cubin")
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
            cur.append([])  # GMMA without a preceding ARRIVE in this function
        cur[-1].append("gsb0" in line)

bad = False
for name, chains in funcs.items():
    chains = [c for c in chains if c]
    n = sum(len(c) for c in chains)
    if not n:
        continue
    flags = []
    if any(any(c[:-1]) for c in chains):
        flags.append("gsb0_not_last")
    if n > 1 and all(len(c) == 1 for c in chains):
        flags.append("serialized")
    if a.min_chain and any(len(c) < a.min_chain for c in chains):
        flags.append("short_chain")
    bad |= bool(flags)
    print(json.dumps({"function": name, "gmma": n, "chains": [len(c) for c in chains],
                      "gsb0": sum(sum(c) for c in chains), "flags": flags}))
sys.exit(1 if bad else 0)
