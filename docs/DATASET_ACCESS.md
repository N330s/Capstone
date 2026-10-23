# Obtain and use the dataset

The dataset is **not stored in GitHub or the Docker image**. Keep raw recordings,
LeRobot exports, downloaded archives and model weights outside Git. Training
scripts, input contracts, hashes and setup instructions are versioned here.

## Recommended: transfer the existing verified release

Ask the project owner/data-collection teammate for the frozen release used by
the native baseline, not simply their newest recording folder:

- `merged_all_v1/`: complete raw manifest and 101 episode directories, including
  lossless `episode.npz`, `metadata.json`, and any recorded diagnostics.
- `export_all_v1/`: existing LeRobot export, including `openarm_export.json` and
  both `train/` and `validation/` trees, with all parquet/video/meta files.
- The matching collector/robot source bundle and the complete model/tokenizer
  directories listed in [LAB_HANDOFF.md](LAB_HANDOFF.md).

Use an approved private transfer, shared drive or external SSD. Copy the existing
manifest/export **byte for byte**, verify SHA-256 after transfer, and preserve the
originals. The release contains 81 train / 20 validation episodes and 147,499 /
37,104 command frames. No need to transfer old run outputs, preview JPEGs,
compressed archive duplicates or a personal Hugging Face login cache.

Place raw data at `$OPENARM_STORAGE/raw/merged_all_v1/` and the export at
`$OPENARM_STORAGE/datasets/export_all_v1/`. These become `/raw/merged_all_v1`
and `/datasets/export_all_v1` inside the container. The host storage root is
configured in `.env`; model and scene paths are in `configs/lab_smolvla.json`.

## Original Drive sources (fallback, not a newer-release instruction)

The collection teammate supplied these folder links:

- [Source folder 1](https://drive.google.com/drive/folders/1o7FoA8RxfBZbj8uEVfBgFK4apqAVKM_J)
- [Source folder 2](https://drive.google.com/drive/folders/16ISEULIuy_DcsBpVUGwgDlvLjqPhtQBT)
- [Source folder 3](https://drive.google.com/drive/folders/1oiJpIR0N5vQHx2bv8Y5-c-zDBUeLohoH)

Folder contents and permissions can change; current access has not been
requalified. Ask the owner for access or immutable copies if needed. The old
baseline used **`v2c30_a.rar`, `v2c30_b.rar`, `v2c30_c.rar`**, totaling about
14.63 GB compressed. Do not bulk-download every folder: newer recovery/depth/
torque archives are a different release and must not silently replace or mix
with this baseline. Identify original files by name plus owner-provided checksums,
not folder order. The folder list above is not an archive-to-folder mapping.

For public/shared-with-anyone files, use the separate download environment and
the pinned `requirements-download.txt`. Example on Linux/WSL:

```bash
python3.11 -m venv .venv-download
.venv-download/bin/python -m pip install -r requirements-download.txt
.venv-download/bin/python -m gdown --no-cookies --folder --json 1oiJpIR0N5vQHx2bv8Y5-c-zDBUeLohoH
```

Repeat the listing for the other folder IDs and select only the original archives.
After obtaining the file ID from a listing, download a selected file to an
existing approved download directory (replace both placeholders):

```bash
.venv-download/bin/python -m gdown --no-cookies --continue FILE_ID_FROM_LISTING -O /approved/downloads/v2c30_a.rar
```

Repeat for b/c, with the correct IDs/names. `--json` lists files without downloading
their contents; `--continue` resumes partial downloads. See the
[official gdown usage](https://github.com/wkentaro/gdown#usage). Do not disable TLS
verification, commit cookies, bypass Drive permissions, or execute archive contents.

## If rebuilding instead of transferring the frozen export

The existing `scripts/extract_collection_rar.py` checks archive paths, selected
expanded size and the 25 GiB free-space reserve before extraction. Its qualified
backend is Windows bsdtar; **ordinary Linux GNU tar cannot be assumed to read RAR**.
Use the existing verified extraction or qualify a safe RAR backend first. Do not
use an unchecked recursive archive extractor just to bypass this requirement.

Then `scripts/merge_collection_release.py` creates a new immutable view and
whole-episode split; `scripts/prepare_smolvla.py` validates/exports it;
`scripts/verify_smolvla_export.py` checks readback. Commands for a raw merged
release, using fresh destinations:

```bash
python scripts/prepare_smolvla.py --dataset /path/to/merged_all_v1
python scripts/prepare_smolvla.py --dataset /path/to/merged_all_v1 --export /path/to/new_export
python scripts/verify_smolvla_export.py --dataset /path/to/merged_all_v1 --export /path/to/new_export --report /path/to/new_readback.json
```

Use the pinned ML runtime for export/readback. Rebuilt manifests may embed
different absolute source paths, and fresh video encoding can change bytes.
Consequently a rebuilt export is **not automatically the frozen release** even
if episode counts agree. Record its identity, audit source/splits/statistics and
normalization compatibility, then requalify before making a new run configuration.
Do not change the frozen config's hashes merely to bypass failed verification.

## Verify and train with the staged frozen release

After building the container and staging all required data/model/assets:

```bash
docker compose run --rm smolvla python scripts/lab_smolvla.py check
docker compose run --rm smolvla python scripts/verify_smolvla_export.py --dataset /raw/merged_all_v1 --export /datasets/export_all_v1 --report /runs/export_readback_001.json
docker compose run --rm smolvla python scripts/lab_smolvla.py qualify --name native_parity_001
docker compose run --rm smolvla python scripts/lab_smolvla.py smoke --name smoke_b8_w0_001 --batch-size 8 --workers 0 --steps 64
```

Require successful hash/readback/parity/train-save-reload checks before a long
run. See [the lab handoff](LAB_HANDOFF.md) for infrastructure, batch benchmarking,
bounded training and live pickup evaluation. The current model uses scene/wrist
RGB plus measured 16-value state, with eight right-arm action targets. Depth and
torque are not current model inputs. A successful offline smoke test alone does
not establish live pickup or generalization.
