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
        self.assertEqual(e.model.nq,25)
        self.assertEqual(self.obs["state"].shape,(16,))
        # Upstream equality constraints synchronize fingers only.
        import mujoco
        self.assertFalse(np.any(e.model.eq_type==mujoco.mjtEq.mjEQ_WELD))
        self.assertEqual(set(self.obs),{"state","timestamp_s","instruction"})

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

