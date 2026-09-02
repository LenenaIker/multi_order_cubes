"""Minimal skrl SAC training entry point for MOC.

Phase B of the SB3 -> skrl migration (see the migration plan): gets a hand-rolled skrl SAC agent
running end-to-end. Deliberately NOT at feature parity with scripts/sb3/train.py yet -- no
--checkpoint resume, no --video, no env.extras -> TensorBoard logging (skrl has no SB3-style
callbacks; that needs a custom training loop instead of the built-in SequentialTrainer used here,
see scripts/sb3/train.py's IsaacInfoTensorboardCallback for what to port), no VecNormalize
equivalent (skrl's own answer is a `state_preprocessor` in cfg/skrl_sac.yaml, not wired up here).
Each of those is its own small follow-up step once this skeleton is validated to actually train.
"""

import argparse
import datetime
from pathlib import Path

import yaml

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC with skrl on multi_order_cubes.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--cfg", type=str, default="cfg/skrl_sac.yaml")
    parser.add_argument(
        "--logdir",
        type=str,
        default="/media/lenena-iker/Crucial/logs/multi_order_cubes/skrl/multi_order_cubes_sac",
    )
    parser.add_argument("--run_name", type=str, default=None)

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    from skrl.agents.torch.sac import SAC
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    from models import build_models

    from multi_order_cubes.config.ur10_gripper import UR10LongSuctionMultiOrderCubesEnvCfg

    set_seed(int(args.seed))

    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{run_stamp}"
    log_root = Path(args.logdir)
    log_root.mkdir(parents=True, exist_ok=True)

    env_cfg = UR10LongSuctionMultiOrderCubesEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = args.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        script_dir = Path(__file__).resolve().parent
        cfg_path = (script_dir.parent.parent / args.cfg).resolve()
    raw_cfg = load_yaml(str(cfg_path))

    net_arch = tuple(raw_cfg.pop("net_arch"))
    memory_size = raw_cfg.pop("memory_size")

    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    models = build_models(env.observation_space, env.action_space, env.device, net_arch)

    agent_cfg = dict(raw_cfg)
    agent_cfg["experiment"] = {
        "directory": str(log_root),
        "experiment_name": run_name,
    }

    agent = SAC(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg=agent_cfg,
    )

    trainer_cfg = {"timesteps": int(args.total_timesteps), "headless": True}
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
