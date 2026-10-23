"""FP32 trainable storage for AMP; preserve checkpoint precision on reload.

Autocast controls computation, not optimizer/master-weight storage. Loading a
SmolVLA expert constructs BF16 parameters even when safetensors stores FP32.
"""
from collections import Counter
from pathlib import Path


def parameter_dtypes(model):
    counts = Counter()
    for parameter in model.parameters():
        if parameter.requires_grad:
            counts[str(parameter.dtype)] += parameter.numel()
    return dict(counts)


def promote_trainable_fp32(model):
    import torch
    before = parameter_dtypes(model)
    for parameter in model.parameters():
        if parameter.requires_grad:
            if not parameter.is_floating_point() or parameter.grad is not None:
                raise ValueError("Promote floating trainable parameters before optimizer/backward")
            parameter.data = parameter.data.to(dtype=torch.float32)
    return {"before": before, "after": parameter_dtypes(model)}


def restore_checkpoint_trainable_dtypes(model, checkpoint):
    """Undo silent from_pretrained downcasting, without modifying frozen VLM."""
    import torch
    from safetensors import safe_open
    checkpoint = Path(checkpoint)
    path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    restored = 0
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name not in tensors.keys():
                raise ValueError("Missing trainable checkpoint tensor: " + name)
            dtype = tensors.get_slice(name).get_dtype()
            target_dtype = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}.get(dtype)
            if target_dtype is None or tuple(tensors.get_slice(name).get_shape()) != tuple(parameter.shape):
                raise ValueError("Unsupported checkpoint parameter dtype/shape")
            if parameter.dtype != target_dtype:
                if parameter.grad is not None:
                    raise ValueError("Restore storage dtype before optimizer/backward")
                value = tensors.get_tensor(name)
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite checkpoint parameter")
                parameter.data = value.to(device=parameter.device, dtype=target_dtype)
                restored += parameter.numel()
    return {"restored_parameters": restored, "trainable_dtypes": parameter_dtypes(model)}


def assert_fp32_optimizer(model, optimizer):
    import torch
    trainable = {id(p): p for p in model.parameters() if p.requires_grad}
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if {id(p) for p in optimized} != set(trainable):
        raise ValueError("Optimizer parameters differ from trainable set")
    for parameter in optimized:
        if parameter.dtype != torch.float32 or (parameter.grad is not None and parameter.grad.dtype != torch.float32):
            raise ValueError("AMP training requires FP32 trainable weights/gradients")
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = optimizer.state.get(parameter, {}).get(key)
            if value is not None and value.dtype != torch.float32:
                raise ValueError("Optimizer moments must be FP32")
