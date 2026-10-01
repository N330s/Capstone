"""Trace align + insert of the tabletop expert on one scene: where does the lateral offset come from?

Prints every 10 steps in align/insert: socket-frame plug offset (y,z), depth, commanded forward,
axial / wall / leaf force, leaf contact count, stall timer, plus the align error at the moment the
expert leaves align.
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import controllers.pick_insert_expert as pie
from envs.openarm_insert import OpenArmInsertEnv
from data_pipeline.scene_bank import as_reset_options

row = json.load(open(sys.argv[1]))
env = OpenArmInsertEnv(images=False)
env.reset(seed=row["seed"], options=as_reset_options(row["options"]))
expert = pie.PickInsertExpert(env, rng=np.random.default_rng(row["seed"]), detector_mode="privileged")
i, last = 0, None
while env.outcome is None and expert.failure is None and i < 6000:
    action = expert.action()
    phase = expert.phase
    _, _, _, _, info = env.step(action)
    p = expert.place
    if p is not None and phase in ("align", "insert", "hold"):
        if phase != last or i % 10 == 0:
            entry = env.data.site_xpos[env.socket_site]
            basis = env.data.site_xmat[env.socket_site].reshape(3, 3)
            local = basis.T @ (env.data.xpos[env.plug] - entry)
            print(f"{i:5d} {phase:6s} t={p.phase_time:5.2f} fwd={getattr(p,'forward',0)*1e3:7.2f} "
                  f"depth={info['insertion_depth_m']*1e3:6.2f} off_y={info['offset_y_m']*1e6:5.0f} off_z={info['offset_z_m']*1e6:5.0f} "
                  f"plug_local_yz={local[1]*1e6:5.0f}/{local[2]*1e6:5.0f}um ori={info['orientation_error_deg']:.2f} "
                  f"axial={info['socket_force_x_n']:6.2f} wall={info['contact_force_n']:5.2f} leaf={info['leaf_contact_force_n']:5.2f} "
                  f"nleaf={info['leaf_contact_count']} stall={getattr(p,'stall_s',0):.2f} eng={getattr(p,'engaged',None)} "
                  f"slip={info['grasp_slip_m']*1e3:.3f} cable={info['cable_force_n']:.2f}", flush=True)
    last, i = phase, i + 1
print("outcome", env.outcome, "failure", expert.failure, expert.failure_detail)
