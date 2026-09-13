"""Replay full-task motor targets, without expert/IK/path-planner decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
from envs.openarm_insert import OpenArmInsertEnv
from controllers.table_pickup import PickupProbe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,default=Path('results/table_insert_validation/nominal'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or args.episode/'replay_report.json'
    if output.exists():raise FileExistsError(output)
    report=json.loads((args.episode/'report.json').read_text())
    rows=json.loads((args.episode/'command_trace.json').read_text())
    if not report['passed'] or report['lower_fixture']:raise ValueError('Expected successful original-height rotated fixture')
    for source,digest in report['source_hashes'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed source: {source}')
    env=OpenArmInsertEnv(images=False,timestep=report['timestep_s'])
    env.model.body_quat[env.model.body('socket').id]=[np.sqrt(.5),np.sqrt(.5),0,0]
    probe=PickupProbe(env)
    errors=[]
    try:
        probe.reset((.381+report['offset_x_mm']/1000,-.22+report['offset_y_mm']/1000,.330))
        settle=[r for r in rows if r['phase']=='settle']
        if not settle:raise ValueError('Missing reset trace')
        np.testing.assert_array_equal(env.data.qpos,np.asarray(settle[-1]['qpos']))
        np.testing.assert_array_equal(env.data.qvel,np.asarray(settle[-1]['qvel']))
        for row in rows[len(settle):]:
            env.target[:]=row['command']
            for _ in range(env.substeps):env._physics_step()
            errors.append([float(np.max(np.abs(env.data.qpos-row['qpos']))),
                           float(np.max(np.abs(env.data.qvel-row['qvel'])))])
            if abs(env.data.time-row['time_s'])>1e-9:raise AssertionError('Timing drift')
            # Match the acquisition loop's command-boundary forward pass. It
            # refreshes bias forces used by the next motor-control interval.
            mujoco.mj_forward(env.model,env.data)
        qpos,qvel=np.max(errors,axis=0)
        result={'passed':bool(qpos<1e-8 and qvel<1e-7),'max_qpos_error':qpos,'max_qvel_error':qvel,
                'commands_after_settle':len(errors),'expert_rerun':False,
                'trace_sha256':hashlib.sha256((args.episode/'command_trace.json').read_bytes()).hexdigest()}
        output.write_text(json.dumps(result,indent=2));print(result)
    finally:env.close()
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
