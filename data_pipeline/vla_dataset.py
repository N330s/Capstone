"""Framework-neutral training samples; simulator diagnostics cannot enter inputs."""
import json
from pathlib import Path
import numpy as np
from data_pipeline.episodes import action_chunk,validate_episode


def prepare_dataset(root):
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text())
    states,actions=[],[]
    for episode in manifest["episodes"]:
        path=root/episode["episode"]
        validate_episode(path)
        if episode["split"]=="train":
            with np.load(path/"episode.npz",allow_pickle=False) as data:
                states.append(data["observation_state"][:-1])
                actions.append(data["action_applied"])
    stats={"schema":manifest["schema"],"fit_split":"train","transforms":{},
           "constant_dimension_rule":"scale=1 when training std < 1e-6; no invented variance"}
    for name,values in (("state",states),("action",actions)):
        array=np.concatenate(values)
        mean,std=array.mean(0),array.std(0)
        scale=np.where(std<1e-6,1.,std)
        stats["transforms"][name]={"mean":mean.tolist(),"scale":scale.tolist()}
    (root/"normalization.json").write_text(json.dumps(stats,indent=2))
    return stats


class PilotVLADataset:
    def __init__(self,root,split="train",horizon=10):
        self.root=Path(root)
        self.horizon=horizon
        manifest=json.loads((self.root/"manifest.json").read_text())
        self.stats=json.loads((self.root/"normalization.json").read_text())
        self.index=[]
        for episode in manifest["episodes"]:
            if episode["split"]==split:
                self.index.extend((episode["episode"],i) for i in range(episode["steps"]))
        self.instruction=manifest["instruction"]

    def __len__(self):
        return len(self.index)

    def normalize(self,key,value):
        stat=self.stats["transforms"][key]
        return (value-np.asarray(stat["mean"]))/np.asarray(stat["scale"])

    def denormalize_action(self,value):
        stat=self.stats["transforms"]["action"]
        return value*np.asarray(stat["scale"])+np.asarray(stat["mean"])

    def __getitem__(self,index):
        episode,step=self.index[index]
        path=self.root/episode
        with np.load(path/"episode.npz",allow_pickle=False) as data:
            state=data["observation_state"][step].copy()
            images={name:data[f"image_{name}"][step].copy() for name in ("scene_rgb","wrist_rgb")}
        actions,mask=action_chunk(path,step,self.horizon)
        return {"images":images,"state":self.normalize("state",state),
                "instruction":self.instruction,"actions":self.normalize("action",actions),
                "action_mask":mask,"episode":episode,"step":step}

