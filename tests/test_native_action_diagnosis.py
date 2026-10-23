import unittest
import numpy as np
from scripts.diagnose_native_actions import errors


class ActionDiagnosisMetricTests(unittest.TestCase):
    def test_exact_commands_have_zero_error(self):
        target = np.arange(400).reshape(50,8)/1000
        for result in errors(target,target).values():
            self.assertEqual(result, {'arm_rmse_rad':0.,'finger_rmse_mm':0.,'finger_bias_mm':0.})

    def test_arm_and_finger_units_are_not_mixed(self):
        target = np.zeros((50,8)); predicted = target.copy()
        predicted[:,:7] = .01; predicted[:,7] = -.002
        for result in errors(predicted,target).values():
            self.assertAlmostEqual(result['arm_rmse_rad'],.01)
            self.assertAlmostEqual(result['finger_rmse_mm'],2.)
            self.assertAlmostEqual(result['finger_bias_mm'],-2.)

    def test_prefix_five_does_not_include_unexecuted_tail(self):
        target = np.zeros((50,8)); predicted = target.copy(); predicted[5:,7] = .01
        result = errors(predicted,target)
        self.assertEqual(result['5']['finger_rmse_mm'],0.)
        self.assertAlmostEqual(result['50']['finger_rmse_mm'],np.sqrt(.9)*10)


if __name__=='__main__': unittest.main()
