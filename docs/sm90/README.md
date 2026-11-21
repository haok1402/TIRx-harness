# SM90 (H200) design notes

- `sm90-plan.md`: the ordered plan for SM90 support across tvm-tirx, TIRx-kernels and TIRx-harness
  (2026-11-21; canonical copy: `kernel-evolve/docs/sm90-plan.md`). Evidence: `sm90-probe/logs/plan/`.
- `sm90-codegen.md`: phase-1 evidence report on SM90 code generation in tvm-tirx (canonical copy:
  `kernel-evolve/docs/sm90-codegen.md`).
- `sm90-probe/`: the probes, run scripts and logs both documents cite. Paths such as
  `sm90-probe/logs/...` resolve here. The full generated `out/` (all `.cu`/PTX/SASS) and
  `upstream_sm90*/` test copies are regenerable with the commands in the appendices and live
  in `kernel-evolve/sm90-probe/`. Large JUnit XML and pytest logs are gzipped.
