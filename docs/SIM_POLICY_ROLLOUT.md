# Trained SmolVLA -> matching simulation (2026-10-04)

The dataset/export/training/checkpoint/inference/simulation path executes. The
tiny update-24 model is **not a useful task policy**: it collides before pickup.
No RL, hardware commands or task-generalization claim is included.

## What was checked

`scripts/rollout_smolvla.py` imports the environment from a separate, unmodified
`automateData` checkout at `b7297ba82701fe7e7814c32ea617a85f832b74a4`, under
`third_party/capstone_automateData`. All ten recording source hashes, the workspace
hash and robot XML hash match `test_collection_new_env/train_00003/metadata.json`.
The runtime scene, action/state order and controller/contact configuration also
match. The initial 16-value state is identical to the recording. Robot meshes
were copied from the existing pinned OpenArm v1 asset bundle; this check hashes
the robot XML, not every mesh individually.

This scene uses the teammate's Type O plug, universal_th socket, flexible cable
and appliance, not the older main-repository two-blade scene. Both OpenArm v1
arms are present; the right acts and the left remains parked. Do not change scene
geometry or reset pose behind recorded training data without a new version.

`learning/vla/smolvla_policy.py` reloads the trained model and saved preprocessing/
postprocessing. Only state, two RGB views and instruction pass the shared adapter.
The postprocessor decodes physical-unit 50x8 chunks; no second normalization is
applied. The evaluator executes five targets then observes again through the
existing environment bounds/rate limits and abort guards. No expert fallback,
phase-dependent controller switching, plug attachment or in-flight qpos writes.

For an untouched base-model comparison, explicitly pass `--base-export` pointing
to the qualified LeRobot export. The original base model is configured for a
different 6-action robot; this option adapts its task features and train-only
normalization exactly as SFT does, without changing its weights. Never use it
with an already-adapted checkpoint, which must use its saved processors. Missing
adaptation fails the feature contract before inference, rather than relaxing the
finite 50x8 action check. Rollout reports include processor-source provenance.

| Check | Trained policy | Recorded expert control |
| --- | --- | --- |
| Commands executed | 12 / 0.235 s | 100 / 2.0 s |
| Outcome | Robot collision; stopped before grasp | Step budget exhausted, no robot collision |
| Peak unwanted robot contact | 65.13 N | 0 N |
| Clipped targets | 12 | 0 |
| Reset state error | 0 | 0 |
| Replay qpos/qvel error | Not a replay | Exactly 0 / 0 |

Policy evidence: [report](../results/smolvla_rollout/heldout_v2/report.json),
[animation](../results/smolvla_rollout/heldout_v2/rollout.gif).
Expert evidence: [report](../results/smolvla_rollout/expert_replay_v1/report.json),
[animation](../results/smolvla_rollout/expert_replay_v1/rollout.gif).
Both runs also save the manifest, requested/applied actions, qpos/qvel, states,
executed durations, diagnostics and chunk logs. Earlier `heldout_v1` is retained.
`pipeline_passed` means commands executed without numerical failure; it does not
mean the policy is safe or succeeds. The newer report has an explicit safety-abort
flag and robot-specific contacts, separate from unrelated cable contacts.

The final collision was between the right hand's left finger and the socket
frame, not plug insertion. The first predicted joint targets differ from the
stationary expert targets by up to 1.38 rad; rate limiting reduces each applied
joint change to at most 0.01 rad but cannot guarantee collision avoidance.
Exact expert replay supports the interface/scene match; it does not prove why
the learned model chose those commands. The 24-update, 12-window diagnostic
provides very little training, especially for initial table-start behavior.

Mean warm model-plus-postprocessing latency was 0.304 s; first prediction 0.841 s.
These exclude observation preprocessing, rendering and model load. None fit the
0.10 s prefix. Simulation pauses while predicting: this is **not real-time**.
Profile on the university 4090 and implement/test a timed policy-worker/controller
handoff before hardware deployment; increasing prefixes without contact-response
checks is not a timing fix.

There are also roughly 3 mm cable/appliance/floor overlaps with large contact
forces in the unchanged recorded scene, including expert replay. They are distinct
from the robot/socket failure and are not covered by the reported plug/socket
penetration metric. Ask the scene owner to review this before contact-physics
calibration or sim-to-real transfer; do not silently repair the recording's scene.

## Reproduce

Use the qualified `.venv-smolvla` with MuJoCo 3.11.0, CUDA PyTorch 2.10.0+cu128
and LeRobot 0.4.4. Source/model/data bundles must be staged first. The separate
checkout must contain `third_party/openarm_mujoco/v1`; a fresh process is required
to avoid importing the older main-repository environment. Missing or mismatched
sources fail verification rather than falling back.

```powershell
$env:HF_HOME = 'C:/Users/Admin/plug_env/data/hf_cache'
.venv-smolvla/Scripts/python.exe -u scripts/rollout_smolvla.py --snapshot third_party/capstone_automateData --episode test_collection_new_env/train_00003 --checkpoint outputs/smolvla_qualification_v2/final_checkpoint/pretrained_model --backbone data/models/smolvlm_backbone --output results/smolvla_rollout/policy_repeat --max-steps 100 --prefix 5
```

Repeat with a different output and `--mode replay` for the recorded-applied-command
control. Replay still uses ordinary physics/controller steps, never qpos playback.
Use a fresh output directory; existing results are refused. Model downloads and
Hub uploads are disabled. The checkpoint weight hash is recorded in the report.

Four lightweight provenance tests reject changed source/workspace/robot XML and
source/workspace path escapes; all 42 repository tests pass. These are separate
from the actual GPU/simulator runs above.

## Next

2026-10-21 update: see [CADENCE_DIAGNOSIS.md](CADENCE_DIAGNOSIS.md) for the
current baseline direction. New policy runs save exact initial raw actor inputs
in `first_actor_input.npz` and per-inference raw state/RGB/instruction hashes
in `chunks.json`; `report.json` binds the first-input file/helper hashes.
These are observation-only evidence writes, never substitutions for live inputs.
Exact simulator reset state alone does not prove that rendered RGB is identical.

Qualify the upstream supervised trainer with full-frame coverage, held-out loss,
process-level resume and bounded checkpoints. Then improve the baseline with
approximately 50 varied full-task demonstrations. Preserve the validation split;
this reset already participated in loss inspection and is not an untouched
benchmark. Build seeded base/trained/expert comparisons on reserved resets before
RLT feature/token training or any sim-to-real claim.
