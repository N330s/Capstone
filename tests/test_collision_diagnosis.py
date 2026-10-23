import unittest
import numpy as np
from learning.vla.collision_diagnosis import first_index, orientation_error_deg, contact_chronology, nearest_path


class CollisionDiagnosisTests(unittest.TestCase):
    def test_one_finger_object_motion_is_not_grasp(self):
        ds=[{'plug_finger_contacts':c,'grasp_established':False} for c in (0,1,1,0)]
        states=np.zeros((4,16)); states[:,15]=[.035,.034,.030,.002]
        r=contact_chronology(ds,states,[.3,.31,.325,.3])
        self.assertEqual(r['first_one_or_two_contacts'],1)
        self.assertIsNone(r['first_two_contacts']); self.assertIsNone(r['first_grasp'])
        self.assertAlmostEqual(r['maximum_plug_lift_mm'],25)
        self.assertEqual(r['first_below_30mm'],3)

    def test_nearest_path_does_not_confuse_phase_lead_with_offpath(self):
        distances,indices=nearest_path([[1,0,0],[2,.01,0]],[[0,0,0],[1,0,0],[2,0,0]])
        np.testing.assert_array_equal(indices,[1,2]); np.testing.assert_allclose(distances,[0,.01])

    def test_orientation_uses_rotation_not_euler_difference(self):
        z=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        self.assertAlmostEqual(float(orientation_error_deg(z,np.eye(3))),90)
        self.assertIsNone(first_index([False,False]))
