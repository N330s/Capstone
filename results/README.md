# Local SmolVLA evidence locations

New plots, videos, metrics CSVs and raw evidence remain on the development PC;
they are **not included in this GitHub handoff**. The paths below are local
locations, not downloadable repository links. Previously published older baseline
reports are unchanged. Start with [current status](../docs/TRAINING_STATUS.md).

| Local folder under `results/` | Interpretation |
| --- | --- |
| `native_smolvla_20261021_v2/` | 12,243 finite native updates; fixed-window validation, not task success |
| `native_pickup_eval_20261022_v2/` | Full81 actor 0/5 live training-reset pickups; expert replay 5/5; held-out not run |
| `native_action_diagnosis_20261022_v2/` | Selected generated-action probes; not a live rollout score |
| `native_single_episode_20261022_v2/` | One live training-reset pickup after single-episode overfit; not generalization |

The actual positive-control video is local at
`outputs/native_single_episode_20261022_v2/training_train_20002_policy/rollout.mp4`.
Videos use simulated time; synchronous inference pauses are omitted. Dataset and
weights transfer separately using [the dataset guide](../docs/DATASET_ACCESS.md)
and [lab handoff](../docs/LAB_HANDOFF.md).
