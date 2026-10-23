import unittest
from learning.vla.native_evaluation import TRAIN_PARENTS,training_gate


class NativeEvaluationGateTests(unittest.TestCase):
    def rows(self):
        return [{'group':'training','episode':parent,'mode':mode,
                 'pickup_success':True,'safety_abort':False}
                for mode in ('replay','policy') for parent in TRAIN_PARENTS]

    def test_requires_all_five_experts_and_all_five_live_pickups(self):
        rows=self.rows()
        self.assertTrue(training_gate(rows))
        for index in range(len(rows)):
            self.assertFalse(training_gate(rows[:index]+rows[index+1:]))

    def test_no_duplicate_parent_or_safety_abort_can_pass(self):
        rows=self.rows(); rows[-1]['episode']=rows[-2]['episode']
        self.assertFalse(training_gate(rows))
        rows=self.rows(); rows[2]['safety_abort']=True
        self.assertFalse(training_gate(rows))

    def test_any_model_or_expert_pickup_failure_blocks_heldout(self):
        for index in range(10):
            rows=self.rows(); rows[index]['pickup_success']=False
            self.assertFalse(training_gate(rows))

    def test_validation_rows_do_not_relax_or_replace_training_gate(self):
        rows=self.rows()
        rows.append({'group':'validation','episode':'new','mode':'policy',
                     'pickup_success':False,'safety_abort':True})
        self.assertTrue(training_gate(rows))
        self.assertFalse(training_gate(rows[1:]))


if __name__=='__main__': unittest.main()
