# Dataset and SmolVLA training visualizations

Open `dataset_training_report.pdf` (six landscape pages) or `index.html`
(offline gallery). PNGs can be copied directly into PowerPoint or Google Slides.
Every chart uses actual recordings or saved test reports; no new training was run.

## Visualization overview

1. **Dataset:** Four successful automated expert recordings, 7,251 commands.
   Three episodes train; one is validation. Two cameras, bimanual state and
   instruction map to eight right-arm targets. These are simulation demonstrations.
2. **Camera views:** The same timestamps appear in both views. The images are
   actual lossless-source expert frames, not an autonomous learned-policy run.
3. **Coverage:** Full recordings cover pickup through insertion. The initial
   qualification used only four windows per episode, so it is a pipeline check,
   not full-dataset training. Phase labels never enter the actor.
4. **Sample:** One observation predicts 50 future commands. Joint targets are
   absolute radians; gripper travel is metres (shown in mm). Terminal padding
   contributes no loss. Labels are applied commands, not next observed positions.
5. **Training:** Pretrained SmolVLA, train-only normalization, masked stochastic
   flow-matching loss, batch one/BF16/AdamW. Save processors with model weights;
   decode predicted targets through the same guarded simulator controller.
6. **Evidence:** The saved 24-update test has finite losses/gradients, improved
   fixed-noise training/validation loss and an exact three-update resume check
   on the same runtime. Raw stochastic loss is not expected to be monotonic.
   The learned policy still collides before grasp: we have proved execution of
   the training/inference pipeline, not task competence or bug-free long runs.

## Sources and reproducibility

- Raw data: `test_collection_new_env/train_00000` through `train_00003`.
- Explicit split: `configs/smolvla_collection_smoke.json`.
- Training: `results/smolvla_training/qualification_v2.json` and `smoke_v3.json`.
- Policy rollout: `results/smolvla_rollout/heldout_v2/report.json`.
- Environment: automateData revision b7297ba82701fe7e7814c32ea617a85f832b74a4,
  matched to recorded source/workspace/robot XML and runtime scene contracts.
- `evidence.json`: episode checksums, exact storyboard/sample frame indices and
  hashes of the selected input reports/config.

Regenerate with a fresh directory:

```powershell
python scripts/visualize_smolvla_dataset.py --output results/dataset_visualization_repeat
```

Requires NumPy, Matplotlib and Pillow in the plotting environment. The command
only reads the original recordings/reports and refuses existing output paths.
