"""Save robot repeatability, recovery and half-timestep acceptance evidence."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from envs.openarm_insert import OpenArmInsertEnv
from controllers.expert import InsertionExpert


def rollout(env, options=None, probe=0.):
    env.reset(options=options)
    expert=InsertionExpert(env,probe_offset_y_m=probe)
    maximum_slip=maximum_park=0.
    while env.outcome is None:
        _,_,_,_,info=env.step(expert.action())
        maximum_slip=max(maximum_slip,info["grasp_slip_m"])
        maximum_park=max(maximum_park,info["left_park_error_rad"])
    return {"outcome":env.outcome,"duration_s":env.elapsed,"depth_m":info["insertion_depth_m"],
            "peak_force_n":env.peak_force,"peak_penetration_m":env.peak_penetration,
            "peak_robot_contact_n":env.peak_robot_contact,"max_slip_m":maximum_slip,
            "max_park_error_rad":maximum_park,"retries":expert.retries}

def main():
    env=OpenArmInsertEnv(images=False)
    aligned=[]
    for i in range(20):
        aligned.append(rollout(env))
        print("aligned",i+1,aligned[-1]["outcome"],flush=True)
    signed=[]
    for field in ("offset_y_m","offset_z_m"):
        for value in (-.001,-.0005,.0005,.001):
            result=rollout(env,{field:value})
            signed.append({"options":{field:value},**result})
            print(field,value,result["outcome"],flush=True)
    recovery=rollout(env,probe=.001)
    fine=OpenArmInsertEnv(images=False,timestep=.00025)
    half=rollout(fine)
    checks={"20_aligned":all(r["outcome"]=="success" for r in aligned),
            "repeatable":all(r==aligned[0] for r in aligned),
            "signed_expert_correction":all(r["outcome"]=="success" for r in signed),
            "recovery":recovery["outcome"]=="success" and recovery["retries"]>0,
            "half_timestep_success":half["outcome"]=="success",
            "half_timestep_depth":abs(half["depth_m"]-aligned[0]["depth_m"])<.0001,
            "half_timestep_force":abs(half["peak_force_n"]-aligned[0]["peak_force_n"])<.2,
            "parked_arm":max(r["max_park_error_rad"] for r in aligned+signed)<.001,
            "bounded_slip":max(r["max_slip_m"] for r in aligned+signed)<.001,
            "no_robot_collisions":all(r["peak_robot_contact_n"]<.01 for r in aligned+signed)}
    report={"checks":checks,"aligned":aligned,"signed":signed,"recovery":recovery,
            "half_timestep":half,"manifest":env.manifest()}
    root=Path("results/openarm_v1")
    root.mkdir(parents=True,exist_ok=True)
    (root/"validation.json").write_text(json.dumps(report,indent=2))
    env.close()
    fine.close()
    print(checks,flush=True)
    raise SystemExit(0 if all(checks.values()) else 1)

if __name__=="__main__":
    main()

