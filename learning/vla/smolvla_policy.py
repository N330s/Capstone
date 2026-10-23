"""Saved SmolVLA processors and one shared physical-unit inference boundary."""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from learning.vla.smolvla_data import (
    observation, decoded_actions, CONTRACT, LEROBOT_VERSION, STATE_NAMES,
    ACTION_NAMES, CAMERAS, sha256,
)


def validate_openarm_features(config):
    """Fail before inference rather than trimming/padding another robot's actions."""
    inputs, outputs = config.input_features, config.output_features
    expected_inputs = {"observation.state", *CAMERAS}
    if (set(inputs) != expected_inputs or set(outputs) != {"action"}
            or tuple(inputs["observation.state"].shape) != (16,)
            or tuple(outputs["action"].shape) != (8,)
            or config.chunk_size != 50):
        raise ValueError("Checkpoint must define OpenArm state=16, action=8, two cameras and chunk=50; "
                         "an unadapted base comparison requires explicit --base-export")
    for camera in CAMERAS:
        shape = tuple(inputs[camera].shape)
        if len(shape) != 3 or shape[0] != 3 or min(shape) <= 0:
            raise ValueError("Expected CHW RGB camera features")


class SmolVLAController:
    def __init__(self, checkpoint, backbone, seed=1000, base_export=None, expert_overlay=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        self.torch = torch
        torch.manual_seed(seed)
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        if cfg.type != "smolvla":
            raise ValueError("Expected SmolVLA checkpoint")
        cfg.vlm_model_name = str(Path(backbone).resolve())
        cfg.load_vlm_weights = False
        cfg.device = "cuda"
        cfg.push_to_hub = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        self.binding = {"processor_source": "saved_checkpoint", "base_export": None}
        if expert_overlay is not None and base_export is not None:
            raise ValueError("Expert overlays require an adapted immutable base, not base-export")
        if base_export is not None:
            # A base-policy baseline needs exactly the same task features and
            # train-only normalization as SFT, with no optimizer updates.
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            export = Path(base_export).resolve()
            receipt_path = export / "openarm_export.json"
            receipt = json.loads(receipt_path.read_text())
            if (receipt["contract"] != CONTRACT or receipt["lerobot_version"] != LEROBOT_VERSION
                    or receipt["state_names"] != STATE_NAMES or receipt["action_names"] != ACTION_NAMES
                    or receipt["camera_mapping"] != CAMERAS):
                raise ValueError("Base adaptation requires the audited OpenArm export contract")
            if (tuple(cfg.input_features["observation.state"].shape) == (16,)
                    and tuple(cfg.output_features["action"].shape) == (8,)):
                raise ValueError("Already-adapted checkpoints must use their saved processors, not --base-export")
            meta = LeRobotDatasetMetadata(receipt["datasets"]["train"]["repo_id"], root=export / "train")
            cfg.input_features, cfg.pretrained_path = {}, Path(checkpoint).resolve()
            self.model = make_policy(cfg, ds_meta=meta)
            self.pre, self.post = make_pre_post_processors(self.model.config, dataset_stats=meta.stats)
            self.binding = {"processor_source": "explicit_train_export_adaptation", "base_export": str(export),
                            "export_receipt_sha256": sha256(receipt_path),
                            "train_stats_sha256": sha256(export / "train/meta/stats.json"),
                            "weights_updated": False}
        else:
            validate_openarm_features(cfg)
            self.model = SmolVLAPolicy.from_pretrained(str(checkpoint), config=cfg)
            from learning.vla.training_precision import restore_checkpoint_trainable_dtypes
            storage = restore_checkpoint_trainable_dtypes(self.model, checkpoint)
            self.binding["trainable_storage"] = storage
            self.pre, self.post = make_pre_post_processors(
                cfg, pretrained_path=str(checkpoint),
                preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name},
                                        "device_processor": {"device": "cuda"}})
        validate_openarm_features(self.model.config)
        if expert_overlay is not None:
            from learning.vla.expert_overlay import load_overlay
            manifest = load_overlay(self.model, checkpoint, expert_overlay)
            self.binding["expert_overlay"] = str(Path(expert_overlay).resolve())
            self.binding["expert_overlay_weights_sha256"] = manifest["weights_sha256"]
        self.model.eval()

    def predict(self, obs, *, noise_seed=None):
        from torch.utils.data import default_collate
        batch = self.pre(default_collate([observation(obs)]))
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            if noise_seed is None:
                prediction = self.model.predict_action_chunk(batch)
            else:
                if type(noise_seed) is not int or not 0 <= noise_seed < 2**63:
                    raise ValueError('Explicit noise seed must be a nonnegative integer')
                noise=self.torch.randn((1,self.model.config.chunk_size,self.model.config.max_action_dim),
                    device='cuda',generator=self.torch.Generator(device='cuda').manual_seed(noise_seed))
                prediction=self.model.predict_action_chunk(batch,noise=noise)
            physical = self.post(prediction)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if tuple(physical.shape) != (1, 50, 8):
            raise ValueError(f"Expected postprocessed 1 x 50 x 8 OpenArm actions, got {tuple(physical.shape)}")
        return decoded_actions(physical[0].detach().cpu().numpy()), elapsed
