import copy
import unittest
from learning.vla.native_single_episode import episode_selection,anchor_state
from learning.vla.smolvla_data import CONTRACT,LEROBOT_VERSION


class NativeSingleEpisodeSelectionTests(unittest.TestCase):
    def receipt(self):
        return {'contract':CONTRACT,'lerobot_version':LEROBOT_VERSION,'datasets':{
            'train':{'episodes':[{'episode':'a','split':'train'},{'episode':'b','split':'train'}]},
            'validation':{'episodes':[{'episode':'v','split':'validation'}]}}}

    def test_returns_native_episode_number_not_global_frame_offset(self):
        receipt=self.receipt(); index,record=episode_selection(receipt,'b')
        self.assertEqual(index,1); self.assertIs(record,receipt['datasets']['train']['episodes'][1])

    def test_validation_and_unknown_parents_rejected(self):
        for name in ('v','absent'):
            with self.assertRaises(ValueError): episode_selection(self.receipt(),name)

    def test_duplicate_or_overlapping_splits_rejected(self):
        for split in ('train','validation'):
            receipt=self.receipt(); receipt['datasets'][split]['episodes'].append(copy.deepcopy(receipt['datasets']['train']['episodes'][0]))
            with self.assertRaises(ValueError): episode_selection(receipt,'a')

    def test_changed_contract_or_release_rejected(self):
        for key in ('contract','lerobot_version'):
            receipt=self.receipt(); receipt[key]='other'
            with self.assertRaises(ValueError): episode_selection(receipt,'a')

    def test_native_one_frame_history_is_accepted_without_mutating_inputs(self):
        import numpy as np
        vector=np.arange(16,dtype=np.float32); native=vector[None,:].copy()
        np.testing.assert_array_equal(anchor_state(native),vector)
        np.testing.assert_array_equal(anchor_state(vector),vector)
        self.assertEqual(native.shape,(1,16))
        for wrong in ((2,16),(1,8),(8,),()):
            with self.assertRaises(ValueError): anchor_state(np.zeros(wrong))


if __name__=='__main__': unittest.main()
