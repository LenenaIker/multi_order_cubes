import gymnasium as gym

# Export the env cfg class so users can import it directly from the package.
from .moc_ur10_env_cfg import UR10LongSuctionMultiOrderCubesEnvCfg

__all__ = ["UR10LongSuctionMultiOrderCubesEnvCfg"]

gym.register(
    id="Isaac-Multi-Order-Cubes-UR10-Long-Suction-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moc_ur10_env_cfg:UR10LongSuctionMultiOrderCubesEnvCfg",
    },
    disable_env_checker=True,
)
