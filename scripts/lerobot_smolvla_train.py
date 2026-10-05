"""Upstream trainer with the explicit LeRobot 0.4.4 SmolVLA mask compatibility fix."""
from importlib.metadata import version


def main():
    if version("lerobot") != "0.4.4":
        raise RuntimeError("Re-audit terminal mask compatibility before changing LeRobot")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    original = SmolVLAPolicy.forward

    def forward(self, batch, *args, **kwargs):
        if "action_is_pad" not in batch:
            raise ValueError("Expected terminal action mask in training batch")
        return original(self, {**batch, "actions_id_pad": batch["action_is_pad"]}, *args, **kwargs)

    SmolVLAPolicy.forward = forward
    from lerobot.scripts.lerobot_train import main as upstream_main
    upstream_main()


if __name__ == "__main__":
    main()
