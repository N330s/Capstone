import random
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
import json

import numpy as np
import torch

from learning.vla.onset_sampling import anchor_partition, sample_indices, chunk_errors, audited_forward, build_index, verify_index
from learning.vla.smolvla_data import CONTRACT, sha256
from learning.vla.training_safety import training_space_available


class OnsetSamplingTests(unittest.TestCase):
    def test_final_only_disk_budget_keeps_same_reserve_without_phantom_periodic_copy(self):
        gib, size = 2**30, 2*2**30
        self.assertFalse(training_space_available(28*gib, size))
        self.assertTrue(training_space_available(28*gib, size, final_only=True))
        self.assertFalse(training_space_available(26*gib, size, final_only=True))
        with self.assertRaises(ValueError):
            training_space_available(100*gib, size, final_only=1)

    def make_fixture(self, root):
        collection, export = root / 'raw', root / 'export'
        folder = collection / 'a'
        folder.mkdir(parents=True); export.mkdir()
        phase = np.array(['survey']*3+['approach']*57+['close']*30)
        np.savez(folder / 'episode.npz', phase=phase,
                 observation_state=np.zeros((91, 16)), action_applied=np.zeros((90, 8)))
        digest = sha256(folder / 'episode.npz')
        (folder / 'metadata.json').write_text(json.dumps({'data_sha256': digest}))
        (export / 'openarm_export.json').write_text(json.dumps({'contract': CONTRACT, 'datasets': {
            'train': {'episodes': [{'episode': 'a', 'split': 'train', 'steps': 90, 'data_sha256': digest}]},
            'validation': {'episodes': [{'episode': 'b', 'split': 'validation'}]}}}))
        return collection, export

    def test_index_is_source_bound_and_keeps_original_probe_indices(self):
        with tempfile.TemporaryDirectory() as d:
            collection, export = self.make_fixture(Path(d))
            index = build_index(collection, export)
            self.assertEqual(index['excluded'], [0, 1, 2])
            self.assertEqual(index['eligible'], list(range(3, 90)))
            active = next(p for p in index['phase_probes'] if p['category'] == 'first_active')
            self.assertEqual(active['index'], 3)
            self.assertEqual(active['step'], 3)
            verify_index(index, collection, export)
            index['eligible'][0] = 0
            with self.assertRaises(ValueError):
                verify_index(index, collection, export)

    def test_changed_source_or_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            collection, export = self.make_fixture(Path(d))
            path = export / 'openarm_export.json'
            receipt = json.loads(path.read_text())
            receipt['datasets']['validation']['episodes'][0]['episode'] = 'a'
            path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                build_index(collection, export)
            receipt['datasets']['validation']['episodes'][0]['episode'] = 'b'
            path.write_text(json.dumps(receipt))
            (collection / 'a/episode.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                build_index(collection, export)

    def test_only_initial_wait_is_removed(self):
        start, eligible = anchor_partition(["survey", "survey", "approach", "survey", "close"])
        self.assertEqual(start, 2)
        self.assertEqual(eligible, [2, 3, 4])
        with self.assertRaises(ValueError):
            anchor_partition(["survey"]*3)

    def test_source_indices_are_not_rebased(self):
        rng = random.Random(3)
        actual = sample_indices(rng, 200, 10, [2, 3, 4, 7, 8, 9])
        self.assertEqual(set(actual), {2, 3, 4, 7, 8, 9})
        self.assertEqual(actual, sample_indices(random.Random(3), 200, 10, [2, 3, 4, 7, 8, 9]))
        expected_rng = random.Random(9)
        expected = [expected_rng.randrange(10) for _ in range(8)]
        self.assertEqual(sample_indices(random.Random(9), 8, 10), expected)

    def test_physical_errors_ignore_terminal_padding(self):
        target = np.zeros((50, 8)); prediction = target.copy()
        prediction[:, 7] = .001; prediction[10:] = 100
        result = chunk_errors(prediction, target, np.arange(50) < 10)
        self.assertAlmostEqual(result["50"]["gripper_mae_mm"], 1)
        self.assertEqual(result["25"]["joint_mae_rad"], [0]*7)

    def test_audit_restores_native_forward_even_on_error(self):
        original = lambda: torch.ones((1, 50, 32))
        policy = SimpleNamespace(model=SimpleNamespace(forward=original))
        batch = {"action_is_pad": torch.zeros((1, 50), dtype=torch.bool)}
        def forward(policy, batch):
            return policy.model.forward().mean(), {}
        loss, _, audit = audited_forward(policy, batch, forward)
        self.assertEqual(float(loss), 1)
        self.assertEqual(audit["actual_dimensions_mean"], 1)
        self.assertIs(policy.model.forward, original)
        def broken(policy, batch):
            raise RuntimeError("test")
        with self.assertRaises(RuntimeError):
            audited_forward(policy, batch, broken)
        self.assertIs(policy.model.forward, original)
