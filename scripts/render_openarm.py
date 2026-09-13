"""Save real simulator camera views and an expert rollout preview."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from envs.openarm_insert import OpenArmInsertEnv
from controllers.expert import InsertionExpert

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("results/openarm_v1/preview"))
    p.add_argument("--probe-offset-y-mm",type=float,default=0)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    env=OpenArmInsertEnv(images=True)
    obs=env.observe()
    for name, pixels in obs["images"].items():
        Image.fromarray(pixels).save(args.output/f"{name}_start.png")
    frames=[]
    expert=InsertionExpert(env,probe_offset_y_m=args.probe_offset_y_mm/1000)
    while not env.outcome:
        obs,_,_,_,info=env.step(expert.action())
        if len(frames)==0 or round(env.elapsed/env.dt)%3==0 or env.outcome:
            panel=Image.new("RGB",(640,240))
            panel.paste(Image.fromarray(obs["images"]["scene_rgb"]),(0,0))
            panel.paste(Image.fromarray(obs["images"]["wrist_rgb"]),(320,0))
            frames.append(panel)
    for name,pixels in obs["images"].items():
        Image.fromarray(pixels).save(args.output/f"{name}_end.png")
    frames[0].save(args.output/"insertion.gif",save_all=True,append_images=frames[1:],duration=60,loop=0)
    (args.output/"summary.json").write_text(json.dumps({k:v for k,v in info.items() if k!="applied_action"},indent=2))
    env.close()
    print(env.outcome)

if __name__=="__main__":
    main()
