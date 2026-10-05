"""Explicit compatibility for the terminal-mask spelling in LeRobot 0.4.4."""


def masked_forward(policy, batch, **kwargs):
    if "action_is_pad" not in batch:
        raise ValueError("Training requires an explicit terminal padding mask")
    adapted = dict(batch)
    # This release reads actions_id_pad, while its dataset emits action_is_pad.
    adapted["actions_id_pad"] = batch["action_is_pad"]
    return policy.forward(adapted, **kwargs)
