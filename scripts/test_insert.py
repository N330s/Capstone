"""Run the shared compliant-holder test; non-success exits with code 1."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from connector.simulation import CONFIG_PATH, ROOT, VERSION, Pose, run_trial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("offset-y-mm", "offset-z-mm", "roll-deg", "pitch-deg", "yaw-deg"):
        p.add_argument("--" + name, type=float, default=0)
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--duration-s", type=float)
    p.add_argument("--speed-m-s", type=float)
    p.add_argument("--timestep", type=float)
    p.add_argument("--output", type=Path, default=ROOT / "results" / VERSION / "single")
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for field in ("duration_s", "speed_m_s"):
        if getattr(args, field) is not None:
            config[field] = getattr(args, field)
    pose = Pose(**{key: getattr(args, key) for key in Pose.__dataclass_fields__})
    result = run_trial(pose, config=config, timestep=args.timestep, output=args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
