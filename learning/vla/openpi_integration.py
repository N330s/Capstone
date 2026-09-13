"""Optional upstream transforms; import only inside the pinned Linux openpi env."""
from learning.vla.openpi_adapter import OpenArmCodec, OpenArmInputs, OpenArmOutputs


def make_data_config(stats, model_config):
    from openpi import transforms
    from openpi.models.model import ModelType
    from openpi.training.config import DataConfig, ModelTransformFactory

    if model_config.model_type != ModelType.PI0 or model_config.action_dim != 32:
        raise ValueError("This audited adapter targets pi0 with 32 model dimensions")
    codec = OpenArmCodec(stats)
    return DataConfig(
        repo_id="local/openarm_v1_pilot", asset_id="openarm_v1_pi0",
        norm_stats={}, use_quantile_norm=False,
        data_transforms=transforms.Group(inputs=[OpenArmInputs(codec)], outputs=[OpenArmOutputs(codec)]),
        model_transforms=ModelTransformFactory()(model_config),
    )


def transformed_pilot(root, stats, model_config, split="train", episode_limit=None):
    """Use the audited upstream transforms without its unfiltered LeRobot loader.

    This is an explicit raw-pilot smoke-test path, not a LeRobot export. The GPU
    trainer must consume this dataset (or an equivalently filtered export).
    """
    from openpi.training.data_loader import transform_dataset
    from learning.vla.pilot_dataset import FullWindowPilotDataset

    raw = FullWindowPilotDataset(root, split, model_config.action_horizon, episode_limit)
    return transform_dataset(raw, make_data_config(stats, model_config))
