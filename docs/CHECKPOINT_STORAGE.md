# Checkpoint storage cleanup

2026-10-20, at the user's request. Removed **four redundant model weight files**,
not entire experiment folders. Reclaimed 4,791,156,896 bytes (4.462 GiB);
free disk increased from 31.801 to 36.263 GiB during cleanup. Disk space had
already recovered from the earlier 23.27 GiB reading before this operation.

| Removed weight copy | Identical weights retained |
| --- | --- |
| Recovery pilot, step 16 best | Same run, step 16 final |
| Recovery sampling / correction, step 400 periodic | Same run, step 400 final |
| Recovery sampling / uniform, step 400 best | Same run, step 400 final |
| Overnight continuation, step 11830 recovery-initial | Original overnight run, step 11830 final |

SHA-256 confirms model, configuration and normalization identity for each pair.
**Every distinct model remains available.** All optimizer/training states,
configuration files, completion receipts, datasets, logs, videos and plots are
preserved. The original pretrained model, original local best, successful FP32
single-parent actor, current five-parent actor and latest FP32 multi-parent
actor were additionally hash-verified before and after cleanup.

The deleted files were removed directly, not sent to the Recycle Bin. They can
be reconstructed exactly by copying the retained identical model file back to
its old path. Each affected `pretrained_model` folder has a generated
`RETIRED_DUPLICATE_MODEL.json` with the source path, target path and hash.
Its historical `complete.json`/best pointer remains historical evidence; that
particular checkpoint copy is not loadable until its weight file is restored.
Use the retained final copy for new work. No unique optimizer state was lost.

Exact paths, retained copies, hashes and deletion status:
`outputs/checkpoint_cleanup_20261020/receipt.json`.
Implementation: `scripts/retire_duplicate_checkpoint_weights.ps1`; default is
read-only, and `-Execute` is a one-time allowlisted operation. It uses native
PowerShell literal-file deletion, refuses path escapes/reparse points and
performs no recursive deletion, hard linking or automatic retries.

This storage maintenance did **not** restart evaluation or training. The cadence
comparison remains stopped with no scored model trials; the previous disk
blocker is now cleared without changing its free-space threshold.
