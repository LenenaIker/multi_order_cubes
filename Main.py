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
from multi_order_cubes.config.ur10_gripper import UR10LongSuctionMultiOrderCubesEnvCfg

from multi_order_cubes.mdp.commands import set_command_from_to, ensure_moc_buffers



def main():
    env_cfg = UR10LongSuctionMultiOrderCubesEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs, info = env.reset()

    # fuerza comando fijo
    set_command_from_to(env, 1, 2)
    ensure_moc_buffers(env)

    for t in range(1000):
        # acciones aleatorias (ajusta a tu action_space)
        actions = torch.randn((env.num_envs, env.action_manager.total_action_dim), device=env.device) * 0.2

        obs, rew, terminated, truncated, info = env.step(actions)

        if bool(terminated.any() or truncated.any()):
            obs, info = env.reset()
            set_command_from_to(env, 1, 2)
            ensure_moc_buffers(env)

    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()
