"""Hash-bound, inference-only trainable-parameter bundles for local diagnostics.

Frozen VLM weights and normalization are reused from an immutable base bundle.
This is a full expert replacement, not LoRA and not an optimizer checkpoint.
"""
import json
from pathlib import Path
import shutil

from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import sha256

FORMAT = "openarm-smolvla-expert-overlay-v1"
FP32_FORMAT = "openarm-smolvla-expert-overlay-v2-fp32"
BASE_FILES = ("model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json",
              "policy_preprocessor_step_5_normalizer_processor.safetensors",
              "policy_postprocessor_step_0_unnormalizer_processor.safetensors")


def verify_binding(manifest, base):
    if manifest.get("format") not in (FORMAT, FP32_FORMAT) or not manifest.get("complete"):
        raise ValueError("Incomplete/unreviewed expert overlay")
    if manifest.get("base_sha256") != {name: sha256(Path(base) / name) for name in BASE_FILES}:
        raise ValueError("Overlay requires the exact original weights/config/normalization")


def save_overlay(model, base, destination):
    import torch
    from safetensors.torch import save_file
    destination, base = Path(destination), Path(base)
    parameters = {name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if p.requires_grad}
    size = sum(p.numel() * p.element_size() for p in parameters.values())
    if not parameters or shutil.disk_usage(destination.parent).free < 25 * 2**30 + size + 8 * 2**20:
        raise RuntimeError("Overlay write would violate the 25 GiB disk reserve")
    destination.mkdir(exist_ok=False)
    save_file(parameters, str(destination / "expert.safetensors"))
    full_precision = all(p.dtype == torch.float32 for p in parameters.values())
    manifest = {"format": FP32_FORMAT if full_precision else FORMAT, "complete": True,
        "trainable_storage": "float32" if full_precision else "legacy mixed precision",
        "base_sha256": {name: sha256(base / name) for name in BASE_FILES},
        "weights_sha256": sha256(destination / "expert.safetensors"),
        "shapes": {name: list(value.shape) for name, value in parameters.items()},
        "parameter_count": sum(p.numel() for p in parameters.values()),
        "tensor_bytes": size, "optimizer_resume_supported": False}
    write_json(destination / "overlay.json", manifest)
    return manifest


def load_overlay(model, base, source):
    import torch
    from safetensors.torch import load_file
    source = Path(source)
    manifest = json.loads((source / "overlay.json").read_text())
    verify_binding(manifest, base)
    if sha256(source / "expert.safetensors") != manifest["weights_sha256"]:
        raise ValueError("Expert overlay weights changed")
    if manifest["format"] == FP32_FORMAT:
        from learning.vla.training_precision import promote_trainable_fp32
        promote_trainable_fp32(model)
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    values = load_file(str(source / "expert.safetensors"))
    if (set(expected) != set(values) or manifest["shapes"] != {k: list(v.shape) for k, v in values.items()}
            or any(expected[k].shape != v.shape or not torch.isfinite(v).all() for k, v in values.items())
            or (manifest["format"] == FP32_FORMAT and any(v.dtype != torch.float32 for v in values.values()))):
        raise ValueError("Expert overlay parameter set/shapes/values differ")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(values[name].to(device=parameter.device, dtype=parameter.dtype))
    return manifest
