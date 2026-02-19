import argparse

from isaaclab.app import AppLauncher

# -----------------
# CLI
# -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -----------------
# Launch SimulationApp FIRST
# -----------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------
# Import Isaac/Omni modules AFTER app is created
# -----------------
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from multi_order_cubes.config.ur10_gripper import (  # noqa: E402
    UR10LongSuctionMultiOrderCubesEnvCfg,
)


def main():
    env_cfg = UR10LongSuctionMultiOrderCubesEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 1000 == 0:
                env.reset()
                print("[INFO] Reset")

            # Safe actions (all zeros)
            actions = torch.zeros_like(env.action_manager.action)
            
            obs, rew, terminated, truncated, info = env.step(actions)
            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
