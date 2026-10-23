"""Small, process-local adapters around the pinned public LeRobot trainer."""
from types import MethodType


def adapt_training_policy(policy, checkpoint):
    from learning.vla.smolvla_policy import validate_openarm_features
    from learning.vla.training_precision import restore_checkpoint_trainable_dtypes, promote_trainable_fp32
    validate_openarm_features(policy.config)
    restore_checkpoint_trainable_dtypes(policy, checkpoint)
    precision = promote_trainable_fp32(policy)
    if getattr(policy, '_openarm_native_mask_adapter', False):
        raise ValueError('Do not install the training adapter twice')
    original = policy.forward

    def forward(self, batch, *args, **kwargs):
        import torch
        mask = batch.get('action_is_pad')
        if mask is None or mask.dtype != torch.bool or tuple(mask.shape) != tuple(batch['action'].shape[:2]):
            raise ValueError('Expected explicit boolean B x H terminal-action mask')
        if 'actions_id_pad' in batch and not torch.equal(batch['actions_id_pad'], mask):
            raise ValueError('Conflicting terminal-mask aliases')
        return original({**batch, 'actions_id_pad': mask}, *args, **kwargs)

    policy.forward = MethodType(forward, policy)
    policy._openarm_native_mask_adapter = True
    return precision


def assert_native_precision(policy, optimizer):
    """Native optimizer may list frozen parameters; they must have no state/grad."""
    import torch
    expected = {id(p) for p in policy.parameters() if p.requires_grad}
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    ids = [id(p) for p in optimized]
    if len(ids) != len(set(ids)) or not expected <= set(ids):
        raise ValueError('Missing/duplicate native optimizer parameters')
    for parameter in optimized:
        if not parameter.requires_grad:
            if parameter.grad is not None or optimizer.state.get(parameter):
                raise ValueError('Frozen backbone received optimizer state/gradients')
            continue
        if parameter.dtype != torch.float32 or (parameter.grad is not None and parameter.grad.dtype != torch.float32):
            raise ValueError('Native trainable storage/gradients must be FP32')
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            value = optimizer.state.get(parameter, {}).get(key)
            if value is not None and value.dtype != torch.float32:
                raise ValueError('Native Adam moments must be FP32')


def local_processors(native_factory, policy_cfg, pretrained_path=None, **kwargs):
    """Keep native saved processors/stats overrides; route tokenizer to staged files."""
    overrides = dict(kwargs.get('preprocessor_overrides') or {})
    overrides['tokenizer_processor'] = {**overrides.get('tokenizer_processor', {}),
                                        'tokenizer_name': policy_cfg.vlm_model_name}
    kwargs['preprocessor_overrides'] = overrides
    return native_factory(policy_cfg, pretrained_path=pretrained_path, **kwargs)


def install_native_adapters(native, emit=lambda event: None):
    """Delegate all optimization to upstream; no site-packages edits."""
    make_policy, make_processors, update = native.make_policy, native.make_pre_post_processors, native.update_policy

    def policy_factory(*args, **kwargs):
        config = kwargs.get('cfg', args[0] if args else None)
        # Draccus recursively merges an empty CLI dict with pretrained features.
        # Derive the embodiment from the actual dataset, not the SO100 base config.
        config.input_features = {}
        policy = make_policy(*args, **kwargs)
        precision = adapt_training_policy(policy, policy.config.pretrained_path)
        emit({'type': 'policy_adapted', 'precision': precision})
        return policy

    def processor_factory(*args, **kwargs):
        return local_processors(make_processors, *args, **kwargs)

    def update_policy(*args, **kwargs):
        policy = kwargs.get('policy', args[1] if len(args) > 1 else None)
        optimizer = kwargs.get('optimizer', args[3] if len(args) > 3 else None)
        accelerator = kwargs.get('accelerator', args[5] if len(args) > 5 else None)
        if accelerator.mixed_precision != 'bf16':
            raise ValueError('Explicit native Accelerator BF16 computation required')
        assert_native_precision(policy, optimizer)
        result = update(*args, **kwargs)
        assert_native_precision(policy, optimizer)
        metrics = result[0]
        emit({'type': 'update', 'step': int(metrics.steps)+1,
              'loss': float(metrics.loss.val), 'grad_norm': float(metrics.grad_norm.val),
              'lr': float(metrics.lr.val), 'update_seconds': float(metrics.update_s.val)})
        return result

    native.make_policy, native.make_pre_post_processors, native.update_policy = policy_factory, processor_factory, update_policy
    return {'make_policy': make_policy, 'make_pre_post_processors': make_processors, 'update_policy': update}
