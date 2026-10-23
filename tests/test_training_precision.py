import tempfile
import copy
from pathlib import Path
import unittest
import torch
from safetensors.torch import save_file

from learning.vla.training_precision import promote_trainable_fp32, restore_checkpoint_trainable_dtypes, assert_fp32_optimizer


class TrainingPrecisionTests(unittest.TestCase):
    def test_small_adam_update_rounds_away_without_master_weights(self):
        low = torch.nn.Parameter(torch.tensor([.02], dtype=torch.bfloat16))
        high = torch.nn.Parameter(low.detach().float().clone())
        before_low, before_high = low.detach().clone(), high.detach().clone()
        for parameter in (low, high):
            optimizer = torch.optim.AdamW([parameter], lr=1e-5, weight_decay=0.)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        self.assertTrue(torch.equal(low, before_low))
        self.assertFalse(torch.equal(high, before_high))

    def test_only_trainable_weights_and_moments_are_promoted(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1)).bfloat16()
        model[1].requires_grad_(False)
        frozen = model[1].weight.detach().clone()
        promote_trainable_fp32(model)
        self.assertEqual(model[0].weight.dtype, torch.float32)
        self.assertEqual(model[1].weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(model[1].weight, frozen))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        model[0](torch.ones(1, 2)).sum().backward()
        optimizer.step()
        assert_fp32_optimizer(model, optimizer)

    def test_loading_restores_fp32_checkpoint_bits_lost_by_constructor(self):
        original = torch.nn.Linear(2, 2).float()
        with torch.no_grad():
            original.weight.fill_(.02001953 + .00001)
        loaded = torch.nn.Linear(2, 2).bfloat16()
        loaded.load_state_dict(original.state_dict())
        self.assertFalse(torch.equal(loaded.weight.float(), original.weight))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            save_file(original.state_dict(), str(path))
            restore_checkpoint_trainable_dtypes(loaded, path)
        self.assertEqual(loaded.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(loaded.weight, original.weight))

    def test_fp32_optimizer_continuation_survives_bf16_constructor(self):
        original = torch.nn.Linear(2, 2).bfloat16()
        promote_trainable_fp32(original)
        optimizer = torch.optim.AdamW(original.parameters(), lr=1e-5)
        x = torch.tensor([[.2, -.4]])

        def update(model, opt):
            opt.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            opt.step()

        update(original, optimizer)
        state = copy.deepcopy(optimizer.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            save_file(original.state_dict(), str(path))
            resumed = torch.nn.Linear(2, 2).bfloat16()
            resumed.load_state_dict(original.state_dict())
            restore_checkpoint_trainable_dtypes(resumed, path)
        opt_resumed = torch.optim.AdamW(resumed.parameters(), lr=1e-5)
        opt_resumed.load_state_dict(state)
        assert_fp32_optimizer(resumed, opt_resumed)
        for _ in range(3):
            update(original, optimizer)
            update(resumed, opt_resumed)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original.parameters(), resumed.parameters())))
