import argparse
from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Play SAC policy with SB3 on multi_order_cubes.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, default=None, help="Path to vecnormalize.pkl (optional).")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize
    from multi_order_cubes.config.ur10_gripper import UR10LongSuctionMultiOrderCubesEnvCfg

    env_cfg = UR10LongSuctionMultiOrderCubesEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = args.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=True)

    if args.vecnormalize is not None:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False

    model = SAC.load(args.checkpoint, env=env, device="cuda" if "cuda" in str(args.device).lower() else "cpu")

    obs = env.reset()
    for i in range(int(args.steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        if i % 20 == 0:
            print(rewards)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()