"""Read a real LeRobot export and compare episode boundaries against raw data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.vla.smolvla_data import CAMERAS, SmolVLASource, LEROBOT_VERSION


def verify(source, export):
    from importlib.metadata import version, distributions
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if version("lerobot") != LEROBOT_VERSION:
        raise ValueError("Unexpected LeRobot version")
    export = Path(export)
    receipt = json.loads((export / "openarm_export.json").read_text())
    if receipt["source_manifest_sha256"] != source.report()["source_manifest_sha256"]:
        raise ValueError("Source manifest differs from export")
    rows = []
    for split in ("train", "validation"):
        entry = receipt["datasets"][split]
        records = [r for r in source.records if r["split"] == split]
        if entry["episodes"] != records:
            raise ValueError("Source selection differs from export")
        ds = LeRobotDataset(entry["repo_id"], root=export / split,
                            delta_timestamps={"action": [i / 50 for i in range(50)]},
                            video_backend="pyav")
        if len(ds) != sum(r["steps"] for r in records):
            raise AssertionError("Export frame count mismatch")
        offset = 0
        for record in records:
            for step in sorted({0, record["steps"] // 2, record["steps"] - 1}):
                expected = source.sample(record["episode"], step)
                actual = ds[offset + step]
                for key in ("observation.state", "action"):
                    np.testing.assert_allclose(actual[key].numpy(), expected[key], atol=1e-6, rtol=0)
                np.testing.assert_array_equal(actual["action_is_pad"].numpy(), expected["action_is_pad"])
                if actual["task"] != expected["task"]:
                    raise AssertionError("Instruction mismatch")
                errors = {}
                for key in CAMERAS:
                    image = actual[key].numpy()
                    if (image.shape != expected[key].shape or not np.isfinite(image).all()
                            or image.min() < 0 or image.max() > 1):
                        raise AssertionError("Bad decoded image")
                    errors[key] = float(np.abs(image - expected[key]).mean() * 255)
                rows.append({"split": split, "episode": record["episode"], "step": step,
                             "valid_actions": int((~expected["action_is_pad"]).sum()),
                             "image_mean_absolute_error_255": errors})
            offset += record["steps"]
    return {"passed": True, "lerobot_version": LEROBOT_VERSION,
            "runtime_versions": {name: version(name) for name in
                                 ("torch", "torchvision", "numpy", "datasets", "av", "transformers")},
            "installed_packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in distributions())),
            "cuda_available": torch.cuda.is_available(),
            "source_manifest_sha256": receipt["source_manifest_sha256"], "samples": rows,
            "limits": "Boundary/midpoint checks only; compression errors reported, visual task quality not certified. No model test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = verify(SmolVLASource(args.dataset, args.selection), args.export)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
