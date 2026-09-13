"""Verify a pilot export and construct train-only normalization plus chunk samples."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from data_pipeline.vla_dataset import prepare_dataset,PilotVLADataset
from data_pipeline.episodes import action_chunk

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=Path("data/openarm_v1_pilot"))
    args=p.parse_args()
    prepare_dataset(args.dataset)
    train=PilotVLADataset(args.dataset,"train")
    val=PilotVLADataset(args.dataset,"validation")
    checks={"nonempty_splits":len(train)>0 and len(val)>0,
            "disjoint_episodes":not ({r[0] for r in train.index}&{r[0] for r in val.index})}
    roundtrip=0.
    for dataset in (train,val):
        for index in (0,len(dataset)-1):
            sample=dataset[index]
            raw,mask=action_chunk(args.dataset/sample["episode"],sample["step"],dataset.horizon)
            roundtrip=max(roundtrip,float(np.max(np.abs(dataset.denormalize_action(sample["actions"])-raw))))
            if not np.array_equal(sample["action_mask"],mask):
                raise AssertionError("Chunk mask mismatch")
    checks["normalization_roundtrip"]=roundtrip<1e-12
    last=train[-1]
    checks["episode_end_masked"]=int(last["action_mask"].sum())==1
    report={"checks":checks,"train_steps":len(train),"validation_steps":len(val),
            "max_action_roundtrip_error":roundtrip,
            "export":"framework-neutral pilot adapter; checkpoint-specific LeRobot export is later"}
    (args.dataset/"training_readiness.json").write_text(json.dumps(report,indent=2))
    print(report)
    raise SystemExit(0 if all(checks.values()) else 1)

if __name__=="__main__":
    main()

