# Cleanup and lab handoff validation

Local host: Windows, RTX 4060 Ti 8 GiB; pinned `.venv-smolvla`, Python 3.11,
PyTorch 2.10.0+cu128, LeRobot 0.4.4. This is **not** lab RTX 4090 qualification.

- Full unittest discovery: 225 tests, **221 passed, four skipped**. Three optional
  plotting tests require the visualization environment; one optional archive-intake
  module requires the separate download environment (`rarfile`). Optional absence
  is now reported as a skip, not an unexplained import failure. No test failure.
- New tests cover portable command planning, input hashes, output isolation,
  batch/budget validation, real elapsed-time throughput, correct output-filesystem
  monitoring, ambiguous GPU rejection, owned-process deadlines and writable
  source archives outside a read-only code tree.
- Real GPU native/reference parity rerun with explicit export/raw/base/backbone
  paths: **two updates matched exactly**, including native loss, clipped gradients
  and weights; frozen backbone unchanged. Completed in 32.547 seconds. No long
  training run or checkpoint deletion was performed.
- Frozen current release checks passed on the actual local inputs: 81 train /
  20 validation, 147,499 train frames, original full81 initializer, training stats,
  raw manifest, matched scene source/workspace/robot XML.
- `scripts/lab_smolvla.py --help` and whitespace checks pass. The local live video
  copy matches its existing policy-trial SHA-256 exactly; no video is uploaded.

Parity report remains local: `outputs/lab_handoff_parity_20261023/report.json`.
Local full logs remain in `outputs/lab_handoff_tests_20261023.log` and
`outputs/lab_handoff_parity_20261023.log`; Windows PowerShell may wrap stderr
messages as `NativeCommandError`, so use the actual unittest result/report rather
than interpreting a warning wrapper as a failed model update.

Still pending on the lab host: Docker image build, full no-skips infrastructure
preflight, native train/save/fresh reload using the new launcher, representative
loader/batch benchmarks, cross-platform expert replay and live positive control.
Do not infer these passes from local unit tests or the parity rerun. See
[the handoff](LAB_HANDOFF.md).
