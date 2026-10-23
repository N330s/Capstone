import copy
import unittest

from learning.vla.five_episode import select_parents, audit_updates
from learning.vla.smolvla_data import CONTRACT, LEROBOT_VERSION


class FiveEpisodeTests(unittest.TestCase):
    def receipt(self):
        return {'contract':CONTRACT, 'lerobot_version':LEROBOT_VERSION, 'datasets': {
            'train':{'episodes':[{'episode':f'train_{i:05d}', 'split':'train'} for i in range(20002,20012)]},
            'validation':{'episodes':[{'episode':'reserved_validation', 'split':'validation'}]}}}

    def test_selection_preserves_control_and_never_uses_validation(self):
        receipt = self.receipt()
        selected = select_parents(receipt)
        self.assertEqual(selected[0]['episode'], 'train_20002')
        self.assertEqual(len({r['episode'] for r in selected}), 5)
        self.assertNotIn('reserved_validation', [r['episode'] for r in selected])
        self.assertEqual(selected, select_parents(receipt))
        receipt['datasets']['validation']['episodes'][0]['episode'] = 'train_20002'
        with self.assertRaises(ValueError):
            select_parents(receipt)

    def fixture(self):
        parents = [{'episode':str(i), 'start':i*100, 'end':i*100+100, 'open_hold_local':[2,3]} for i in range(5)]
        rows = []
        for i in range(10):
            p = parents[i%5]
            rows.append({'step':i+1, 'episode':p['episode'], 'indices':[p['start']+2]*4+[p['start']+99]*4,
                'gripper_sensor_offsets_m':[.001]*4, 'loss':.001, 'grad_norm':.1, 'lr':1e-5})
        return parents, rows

    def test_equal_parent_exposure_and_exact_openhold_quota(self):
        parents, rows = self.fixture()
        audit = audit_updates(rows, parents, 10)
        self.assertEqual(set(audit['updates_per_parent'].values()), {2})
        self.assertEqual(audit['guaranteed_open_hold_anchors'], 40)
        self.assertEqual(audit['nonzero_perturbed_anchors'], 40)

    def test_wrong_parent_wrong_quota_or_unbounded_noise_rejected(self):
        parents, rows = self.fixture()
        for field, value in [('episode','wrong'), ('indices',[99]*8),
                             ('gripper_sensor_offsets_m',[.003]*4), ('loss',float('nan'))]:
            bad = copy.deepcopy(rows); bad[0][field] = value
            with self.assertRaises(ValueError):
                audit_updates(bad, parents, 10)
