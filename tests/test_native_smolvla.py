from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from safetensors.torch import save_file
from learning.vla.native_smolvla import adapt_training_policy, assert_native_precision, local_processors, install_native_adapters


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((8,), .02001953))
        self.frozen = torch.nn.Parameter(torch.ones(8), requires_grad=False)
        feature = lambda shape: SimpleNamespace(shape=shape)
        self.config = SimpleNamespace(chunk_size=50,
            input_features={'observation.state': feature((16,)),
                'observation.images.camera1': feature((3,24,32)),
                'observation.images.camera2': feature((3,24,32))},
            output_features={'action': feature((8,))})

    def forward(self, batch, **kwargs):
        values = batch['action']*self.weight
        return (values*(~batch['actions_id_pad']).unsqueeze(-1)).square().mean(), {}


class NativeAdapterTests(unittest.TestCase):
    def adapted(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        policy = TinyPolicy()
        save_file(policy.state_dict(), str(Path(folder.name)/'model.safetensors'))
        policy.bfloat16()
        adapt_training_policy(policy, folder.name)
        return policy

    def batch(self):
        return {'action': torch.ones(2,50,8), 'action_is_pad': torch.zeros(2,50,dtype=torch.bool)}

    def test_alias_loss_and_fp32_trainable_frozen_precision(self):
        policy = self.adapted()
        batch = self.batch()
        batch['action_is_pad'][:,1:] = True
        loss,_ = policy(batch)
        self.assertEqual(policy.weight.dtype, torch.float32)
        self.assertEqual(policy.frozen.dtype, torch.bfloat16)
        self.assertNotIn('actions_id_pad', batch)
        expected = (batch['action']*policy.weight*(~batch['action_is_pad']).unsqueeze(-1)).square().mean()
        self.assertTrue(torch.equal(loss, expected))

    def test_invalid_or_conflicting_mask_rejected(self):
        policy = self.adapted()
        for batch in ({'action': self.batch()['action']},
                      {**self.batch(), 'action_is_pad': torch.zeros(2,50)},
                      {**self.batch(), 'actions_id_pad': torch.ones(2,50,dtype=torch.bool)}):
            with self.assertRaises(ValueError): policy(batch)

    def test_native_optimizer_can_include_frozen_but_not_update_it(self):
        policy = self.adapted()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        policy(self.batch())[0].backward(); optimizer.step()
        assert_native_precision(policy, optimizer)
        self.assertEqual(policy.frozen.grad, None)
        optimizer.state[policy.frozen]['exp_avg'] = torch.ones_like(policy.frozen)
        with self.assertRaises(ValueError): assert_native_precision(policy, optimizer)

    def test_precision_and_coverage_errors_rejected(self):
        policy = self.adapted()
        optimizer = torch.optim.AdamW([policy.frozen])
        with self.assertRaises(ValueError): assert_native_precision(policy, optimizer)
        policy.weight.data = policy.weight.data.bfloat16()
        with self.assertRaises(ValueError): assert_native_precision(policy, torch.optim.AdamW(policy.parameters()))

    def test_local_tokenizer_preserves_native_stats_overrides(self):
        calls = []
        def factory(*args,**kwargs): calls.append((args,kwargs)); return 'pre','post'
        config = SimpleNamespace(vlm_model_name='staged/backbone')
        kwargs = {'preprocessor_overrides': {'normalizer_processor': {'stats': {'x':1}}}}
        self.assertEqual(local_processors(factory, config, 'base', **kwargs), ('pre','post'))
        self.assertEqual(calls[0][1]['preprocessor_overrides']['normalizer_processor']['stats'], {'x':1})
        self.assertEqual(calls[0][1]['preprocessor_overrides']['tokenizer_processor']['tokenizer_name'], 'staged/backbone')
        self.assertNotIn('tokenizer_processor', kwargs['preprocessor_overrides'])

    def test_factory_clears_recursively_merged_base_features_before_native_adaptation(self):
        with TemporaryDirectory() as folder:
            policy = TinyPolicy()
            policy.config.pretrained_path = Path(folder)
            save_file(policy.state_dict(), str(Path(folder)/'model.safetensors'))
            observed = []
            def factory(cfg, ds_meta): observed.append(dict(cfg.input_features)); return policy
            native = SimpleNamespace(make_policy=factory, make_pre_post_processors=lambda *a,**k:None,
                                     update_policy=lambda *a,**k:None)
            install_native_adapters(native)
            cfg = SimpleNamespace(input_features={'observation.state': 'stale SO100', 'camera3': 'stale'})
            self.assertIs(native.make_policy(cfg=cfg, ds_meta=object()), policy)
            self.assertEqual(observed, [{}])


if __name__ == '__main__': unittest.main()
