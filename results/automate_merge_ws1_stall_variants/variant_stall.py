"""Run the stalled scenes under one insertion variant.

usage: variant_stall.py <variant> <row json>...
  base          current code
  lead1         ENGAGED_LEAD_M = 1 mm (seat target at most 1 mm ahead of the plug once leaves engage)
  lead3         ENGAGED_LEAD_M = 3 mm
  cap20         push cap 15 -> 20 N once engaged
"""
import sys, json, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
import controllers.place_insert as pi
from envs.openarm_insert import OpenArmInsertEnv
import collect_random as cr

variant = sys.argv[1]
if variant.startswith("lead"):
    pi.ENGAGED_LEAD_M = float(variant[4:]) / 1000
elif variant == "cap20":
    orig_init = pi.PlaceInsert.__init__
    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.push_cap = 20.0
    pi.PlaceInsert.__init__ = init

env = OpenArmInsertEnv(images=False)
for path in sys.argv[2:]:
    r = json.load(open(path))
    row = {k: r[k] for k in ('id', 'seed', 'split', 'options')}
    t = time.time()
    summary, _, _ = cr.rollout(env, row, detector_mode="privileged")
    e = summary.get('expert', {})
    pl = e.get('place', {}) or {}
    st = pl.get('stats') or {}
    print(variant, row['id'], summary.get('outcome'), '|', e.get('failure_detail') or '',
          '| depth', round(st.get('final_depth_m', 0) * 1e3, 2), 'peak_wall', round(st.get('peak_socket_force_n', 0), 2),
          '| env peak wall', round(summary.get('final', {}).get('peak_contact_force_n', 0), 2),
          '|', round(time.time() - t), 's', flush=True)
