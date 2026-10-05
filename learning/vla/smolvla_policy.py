"""Saved SmolVLA processors and one shared physical-unit inference boundary."""
import time
from pathlib import Path

from learning.vla.smolvla_data import observation, decoded_actions


class SmolVLAController:
    def __init__(self, checkpoint, backbone, seed=1000):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.torch = torch
        torch.manual_seed(seed)
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        if cfg.type != "smolvla":
            raise ValueError("Expected SmolVLA checkpoint")
        cfg.vlm_model_name = str(Path(backbone).resolve())
        cfg.load_vlm_weights = False
        cfg.device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        self.model = SmolVLAPolicy.from_pretrained(str(checkpoint), config=cfg)
        self.model.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name},
                                    "device_processor": {"device": "cuda"}})

    def predict(self, obs):
        from torch.utils.data import default_collate
        batch = self.pre(default_collate([observation(obs)]))
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            prediction = self.model.predict_action_chunk(batch)
            physical = self.post(prediction)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return decoded_actions(physical[0].detach().cpu().numpy()), elapsed
