# HARNESS-001: what the M3 evolution agent tripped on (rmsnorm_h200, H200, secured kcoral)

Source: the agent's own end-of-session report plus its transcript (`m3/agent.jsonl`). Run: `haok/h200-harness-stack`
`606b741`, task `rmsnorm_h200`, kcoral behind the C4 token proxy, agent in a CPU-only Slurm step.

| # | Finding | Effect | Owner / fix |
|---|---|---|---|
| 1 | The skills' remote tool CLIs (`kcoral_python.py`, `kcoral_ncu.py`, `kcoral_iket.py`, `kcoral_compute_sanitizer.py`) don't accept `--remote @URL_FILE`. Given the plain URL they get HTTP 401: only `kcoral_remote.py` reads and sends the bearer token. | **No IKET, NCU or Compute Sanitizer at all** in a secured remote run. The agent fell back to hand-timed `--cmd` sweeps. | C4 follow-up: give the tool CLIs the same `resolve_url` / `token_for` path (shared helper). |
| 2 | PROMPT_remote.md's `curl …/health` check returns 401 behind the proxy. | A misleading "server down" signal. | C4 follow-up: document `-H "Authorization: Bearer $(cat URL_FILE.token)"`, or add a `kcoral_remote.py --health`. |
| 3 | `--cmd` scripts find no `python` on the server's PATH. The server venv isn't activated in the step. | The agent had to spell out the venv interpreter's full path. | C4 follow-up: `slurm_kcoral.sh` should export the venv's `bin` on PATH for the server step. |
| 4 | The B0 seed's sm_90 path uses `allowed_func_calls` (the packed f32x2 helper); the task's authoring rules forbid that flag. | The agent couldn't reuse the seed as-is and switched to scalar multiplies. | C3 spec: either allow that one helper, or say that the seed's sm_90 path needs it. |
| 5 | The banned-path Bash hook blocked a harmless command. It seemed to match the working directory, which sits under a banned prefix (the run lives inside `m3-stack/`, which is itself a harness checkout). | One blocked command; no data. | Investigate the hook's prefix match for runs whose root is inside a checkout. |
| 6 | The wiki's fetched `tirx-kernels` reference checkout lacks the SM90 kernels (pinned upstream), while the installed (overlaid) package has them, and sm90.md points at the reference paths. | Not reported as blocking: the agent read the installed package. | C6/C7: the dev overlay should also refresh `references/repos/tirx-kernels`, or sm90.md should name the installed path. |

No issue with the SM90 documentation itself was reported. The agent found no misleading compiler remarks: none
of its frontier kernels had spills or C75xx remarks.
