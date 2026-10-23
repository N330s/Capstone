import unittest
import numpy as np
from scripts.diagnose_cadence_repeatability import digest_array
from scripts.report_cadence_repeatability import difference
from learning.vla.inference_inputs import inference_input_fingerprint


class RepeatabilityAuditTests(unittest.TestCase):
    def test_hash_binds_shape_and_dtype_not_only_bytes(self):
        a = np.zeros((2, 3), dtype=np.uint8)
        self.assertNotEqual(digest_array(a), digest_array(a.reshape(3, 2)))
        self.assertNotEqual(digest_array(a), digest_array(a.astype(np.int8)))

    def test_hash_detects_single_intensity_change(self):
        a = np.zeros((240, 320, 3), dtype=np.uint8)
        b = a.copy()
        b[4, 5, 1] = 1
        self.assertNotEqual(digest_array(a), digest_array(b))
        self.assertEqual(digest_array(a), digest_array(a.copy()))

    def test_physical_units_are_reported_separately(self):
        a = np.zeros((50, 8))
        b = a.copy()
        b[0, 0] = .01
        b[2, 7] = .002
        self.assertEqual(difference(a, b),
                         {'max_abs': .01, 'joint_max_abs_rad': .01, 'finger_max_abs_mm': 2.})

    def test_identical_chunk_is_exactly_zero(self):
        a = np.random.default_rng(1000).normal(size=(50, 8))
        self.assertTrue(all(v == 0 for v in difference(a, a.copy()).values()))

    def test_malformed_or_nonfinite_chunks_rejected(self):
        a = np.zeros((50, 8))
        for b in (np.zeros((5, 8)), np.full_like(a, np.nan), np.full_like(a, np.inf)):
            with self.assertRaises(ValueError):
                difference(a, b)

    def observation(self):
        return {'state': np.zeros(16, dtype=np.float32), 'instruction': 'Insert the plug.',
                'images': {k: np.zeros((24, 32, 3), dtype=np.uint8)
                           for k in ('scene_rgb', 'wrist_rgb')}}

    def test_raw_input_audit_covers_each_camera_state_and_instruction(self):
        original = inference_input_fingerprint(self.observation())
        for key in ('scene_rgb', 'wrist_rgb', 'state', 'instruction'):
            obs = self.observation()
            if key == 'instruction': obs[key] += ' now'
            elif key == 'state': obs[key][0] = .001
            else: obs['images'][key][0, 0, 0] = 1
            self.assertNotEqual(original, inference_input_fingerprint(obs))

    def test_audit_does_not_modify_inputs_or_forward_privileged_fields(self):
        obs = self.observation()
        obs['qpos'] = np.ones(148)
        before = {k: v.copy() for k, v in obs['images'].items()}
        fingerprint = inference_input_fingerprint(obs)
        self.assertEqual(set(fingerprint['arrays']), {'state', 'scene_rgb', 'wrist_rgb'})
        for key, value in before.items(): np.testing.assert_array_equal(value, obs['images'][key])

    def test_input_types_are_not_silently_coerced(self):
        obs = self.observation()
        obs['state'] = obs['state'].astype(np.float64)
        with self.assertRaises(ValueError): inference_input_fingerprint(obs)


if __name__ == '__main__':
    unittest.main()
