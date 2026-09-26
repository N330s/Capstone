"""Robot interface tests; run heavier acceptance through validate_openarm.py."""
import unittest
import numpy as np
from envs.openarm_insert import OpenArmInsertEnv
from controllers.chunks import execute_chunk


class OpenArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=OpenArmInsertEnv(images=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.obs,self.info=self.env.reset()

    def test_v1_bimanual_and_no_weld(self):
        e=self.env
        self.assertEqual(len(e.joints["left"]),7)
        self.assertEqual(len(e.joints["right"]),7)
        # 14 arm + 4 finger + 7 plug free joint, plus 4 per cable ball joint and 1 per leaf slide.
        segments=e.workspace["cable"]["segments"]
        self.assertEqual(e.model.nq,25+4*segments+len(e.leaf_joints))
        self.assertEqual(e.model.nv,24+3*segments+len(e.leaf_joints))
        self.assertEqual(self.obs["state"].shape,(16,))
        # Upstream equality constraints synchronize fingers; the cable end is a connect, never a weld.
        import mujoco
        self.assertFalse(np.any(e.model.eq_type==mujoco.mjtEq.mjEQ_WELD))
        self.assertEqual(int(np.sum(e.model.eq_type==mujoco.mjtEq.mjEQ_CONNECT)),1)
        self.assertEqual(set(self.obs),{"state","timestamp_s","instruction"})

    def test_workspace_cable_and_leaves(self):
        e=self.env; m,d=e.model,e.data
        end=d.site_xpos[m.site("cable_end").id]; anchor=d.site_xpos[m.site("cable_anchor").id]
        self.assertLess(np.linalg.norm(end-anchor),.002)
        self.assertEqual(len(e.cable_bodies),e.workspace["cable"]["segments"])
        # Leaves rest against their upper stop under the preload; joint ranges are [-travel, 0].
        for j in e.leaf_joints:
            self.assertLess(abs(float(d.qpos[m.jnt_qposadr[j]])),5e-5)
            self.assertGreater(float(d.qfrc_passive[m.jnt_dofadr[j]]),1.)
        info=self.info
        for key in ("cable_force_n","cable_tension_n","leaf_normal_n","socket_force_x_n",
                    "socket_torque_nm","plug_weight_n","cable_robot_contact_n"):
            self.assertTrue(np.isfinite(info[key]))
        self.assertAlmostEqual(info["plug_weight_n"],.03*9.81,places=3)
        manifest=e.manifest()
        self.assertEqual(manifest["workspace"]["path"],"configs/workspace_v1.json")
        self.assertEqual(manifest["workspace"]["sha256"],e.workspace["_sha256"])
        self.assertIn("envs/workspace.py",manifest["source_hashes"])

    def test_actions_validated_before_advancing(self):
        before=self.env.data.time
        for a in (np.zeros(7),np.full(8,np.nan)):
            with self.assertRaises(ValueError):
                self.env.step(a)
        self.assertEqual(self.env.data.time,before)

    def test_rate_limits_and_no_plug_forces(self):
        e=self.env
        previous=e.target.copy()
        a=previous.copy()
        a[0]+=1
        _,_,_,_,info=e.step(a)
        self.assertLessEqual(info["applied_action"][0]-previous[0],.01000000001)
        self.assertTrue(info["action_clipped"])
        self.assertTrue(np.all(e.data.xfrc_applied==0))
        self.assertTrue(np.all(e.data.qfrc_applied==0))
        self.assertLess(info["robot_unwanted_contact_n"],.01)

    def test_reset_reproduces_controller_state(self):
        e=self.env
        before=e.data.qpos.copy()
        action=e.target.copy()
        e.step(action)
        e.reset()
        np.testing.assert_array_equal(e.data.qpos,before)
        np.testing.assert_array_equal(e.target,action)

    def test_chunk_stops_on_terminal(self):
        class TerminalEnv:
            def __init__(self): self.calls=0
            def step(self,a):
                self.calls+=1
                return {},1,True,False,{}
        e=TerminalEnv()
        self.assertEqual(len(execute_chunk(e,np.zeros((10,8)))),1)
        self.assertEqual(e.calls,1)

if __name__=="__main__":
    unittest.main()

