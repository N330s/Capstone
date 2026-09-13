"""Deterministic multi-start IK audit using the measured physical pickup grasp."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from envs.openarm_insert import OpenArmInsertEnv
from scripts.preview_table_task import inspect_ik


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/transport_audit_v1'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    grasp=json.loads(Path('results/table_pickup_v3/report.json').read_text())['cases'][0]
    relative_rotation=np.array(grasp['grasp_rotation_in_tool'])
    relative_position=np.array(grasp['grasp_position_in_tool_m'])
    env=OpenArmInsertEnv(images=False)
    rng=np.random.default_rng(20260913)
    seeds=[np.array([0,0,0,1.57,0,0,0.])]+[rng.uniform(env.lower+.05,env.upper-.05) for _ in range(24)]
    variants=[('original',[.4391,-.155,.478],0),('original_half_turn',[.4391,-.155,.478],180),
              ('rotated_socket',[.4391,-.155,.478],90),('lower_rotated_socket',[.4191,-.22,.44],90)]
    results=[]
    try:
        for name,position,roll in variants:
            angle=np.radians(roll)
            rotation=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
            tool_rotation=rotation@relative_rotation.T
            tool_position=np.array(position)-tool_rotation@relative_position
            trials=[]
            for seed in seeds:
                _,check=inspect_ik(env,tool_position,tool_rotation,seed)
                trials.append(check)
            valid=[t for t in trials if t['ik_converged'] and not t['unexpected_contacts']]
            best=max(valid,key=lambda t:t['minimum_joint_limit_margin_rad']) if valid else min(trials,key=lambda t:t['position_error_m'])
            results.append({'variant':name,'socket_position_m':position,'socket_roll_deg':roll,
                'valid_solutions':len(valid),'best':best,'trials':trials,
                'note':'Robot endpoint screen only; carried plug and modified fixture collision/path checks are still required'})
            (args.output/'report.json').write_text(json.dumps({'seed':20260913,'results':results},indent=2))
            print(name,len(valid),best['minimum_joint_limit_margin_rad'],best['position_error_m'],flush=True)
    finally:env.close()


if __name__=='__main__':main()
