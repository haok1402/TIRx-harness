# SM90 (H200) design notes

- `sm90-codegen.md`: phase-1 evidence report on SM90 code generation in tvm-tirx (canonical copy:
  `kernel-evolve/docs/sm90-codegen.md`).
- `sm90-probe/`: the probes, run scripts and logs the report cites. Paths in the report such as
  `sm90-probe/logs/...` resolve here. The full generated `out/` (all `.cu`/PTX/SASS) and
  `upstream_sm90/` test copies are regenerable with the commands in the report's appendix and live
  in `kernel-evolve/sm90-probe/`. The broad JUnit XML is gzipped.
