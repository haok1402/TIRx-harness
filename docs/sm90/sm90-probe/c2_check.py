import sys, tempfile
from pathlib import Path
from evolution import cli
from evolution.preparation import target as gt
from evolution.preparation.declare import load_task, load_toolset, task_path
t = gt.detect_local()
print("detected:", t.as_dict(), t.label)
try:
    cli.main(["init", "--task", "kda_forward"])
except SystemExit as e:
    print("evolve init --task kda_forward ->", "exit", e.code)
task = load_task(task_path(cli.REPO_ROOT, "fp16_gemm_floor"))
prompt = cli._prompt(task=task, toolset=load_toolset(cli.REPO_ROOT / "evolution/toolsets/kda_flow.yaml"),
                     worktree=Path(tempfile.mkdtemp()), target=t)
print("fp16_gemm_floor prompt line:", next(l for l in prompt.splitlines() if "FP16 GEMM on" in l))
