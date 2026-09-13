"""Single physical-grasp OpenArm v1 insertion diagnostic."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from envs.openarm_insert import OpenArmInsertEnv
from controllers.expert import InsertionExpert


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offset-y-mm",type=float,default=0)
    p.add_argument("--offset-z-mm",type=float,default=0)
    args=p.parse_args()
    env=OpenArmInsertEnv(images=False)
    env.reset(options={"offset_y_m":args.offset_y_mm/1000,"offset_z_m":args.offset_z_mm/1000})
    expert=InsertionExpert(env)
    for i in range(500):
        _,_,terminated,truncated,info=env.step(expert.action())
        if i%25==0 or terminated or truncated:
            print(i,expert.phase,{k:round(info[k],6) for k in
                ("insertion_depth_m","offset_z_m","orientation_error_deg","contact_force_n","grasp_slip_m")},
                env.outcome,flush=True)
        if terminated or truncated:
            break
    env.close()
    raise SystemExit(0 if env.outcome=="success" else 1)

if __name__=="__main__":
    main()

