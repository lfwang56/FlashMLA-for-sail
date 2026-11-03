# Native FP8 sparse consolidation: validation


Tested commit: `05c31d5a629db4b55ee5db45f32f7f86471add41`.
Baseline: `ccc7cadf3e3a55f5caedd4f3be32df8616eb6329`.

- Reproducible transform and exact-tree comparison: **PASS**. The prefill/decode implementation tokens and original shared components were also checked against the baseline.
- Built both existing targets, `ppu_10` and `ppu_15`, through the project's cudafy compatibility build. Runtime tests were performed on **ZW-M890P / PPU0015** with HGGC `2.2.0-663c22` and PyTorch `2.11.0`.
- `tests/test_sparse_quantized.py`: **860 passed**, including API contracts and graph replay.
- Existing FlashMLA operator matrix: **6325/6325 passed**, with no missing cases, execution errors, or timeouts. Coverage includes native prefill, native/packed decode, MTP/DSpark draft and verify, and dense MLA.

The candidate matrix was rerun in full and compared with the archived main baseline on the same cards, SDK, workload, and unchanged harness source snapshot. Timing retained warmup=2 and repeats=5.

### Performance and numerical comparison

- All **27** single-pass latency alerts above 10% were checked with one full preparation round and five full confirmation rounds. All were classified as not reproduced: **0 confirmed regressions, 0 unstable cases, and 0 unresolved cases**. The performance gate passed.
- The strict relative comparison retained **9 LSE error-growth alerts**, with observed LSE errors below `1e-7`. All original correctness gates passed. For each alert, the unchanged baseline and candidate were each rerun six times; all rounds passed the original correctness gates, and baseline self-runs reproduced all nine relative-growth alerts. The original alerts remain recorded, so the **strict relative gate remains false**.
- Both builds emitted 1740 kernel resource records; 29 register-allocation records changed after consolidation. This is a source-level mechanical refactor, not a claim of identical generated binaries or a speedup.

Validation covers the operator matrix and API/graph tests above. It does not establish PPU001 runtime behavior or full-model serving performance.


The full raw archive is retained privately because it contains internal paths and environment metadata. This public summary reports the observed results, including the non-passing strict relative gate.

The accompanying scripts are the exact tested versions. The verifier requires Python 3.10+, Git identity configuration, and the original workspace parent `/root/lfwang/Projects/t-head/worktrees` to exist. Run from a checkout containing both commits. It leaves a verification worktree and uses branch `codex/verify-sparse-merge-final-ccc7cadf`; an existing branch with that name must be handled explicitly before rerunning. No PPU device is required for tree verification.
