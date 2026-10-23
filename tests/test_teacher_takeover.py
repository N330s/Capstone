import unittest
import numpy as np
from learning.vla.teacher_takeover import check_prefix_state, command_owner,separation_direction


class TakeoverTests(unittest.TestCase):
    def test_prefix_reproduction(self):
        self.assertEqual(check_prefix_state([1.,2.],[0.,0.],[1.,2.],[0.,0.]), {'qpos':0.,'qvel':0.})
        with self.assertRaises(ValueError):
            check_prefix_state([1.001],[0.],[1.],[0.])
        with self.assertRaises(ValueError):
            check_prefix_state([np.nan],[0.],[1.],[0.])

    def test_prefix_is_not_teacher_target(self):
        self.assertEqual(command_owner(274,275), 'prefix_replay')
        self.assertEqual(command_owner(275,275), 'teacher_recovery')
        with self.assertRaises(ValueError):
            command_owner(-1,275)

    def test_separation_is_away_not_toward_plug(self):
        np.testing.assert_allclose(separation_direction([[0.,1.,0.]]),[0.,-1.,0.])
        v=separation_direction([[-.24,.97,0.]])
        self.assertLess(np.dot(v,[-.24,.97,0.]),-.99)
        self.assertEqual(v[2],0.)

    def test_ambiguous_separation_rejected(self):
        for normals in ([],[[0.,0.,1.]],[[1.,0.,0.],[-1.,0.,0.]],[[np.nan,0.,0.]]):
            with self.assertRaises(ValueError): separation_direction(normals)


if __name__ == '__main__': unittest.main()
