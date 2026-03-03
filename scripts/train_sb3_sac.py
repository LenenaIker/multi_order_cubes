import argparse
from pathlib import Path
import datetime
import yaml
import copy

from isaaclab.app import AppLauncher

from stable_baselines3.common.callbacks import BaseCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC with SB3 on multi_order_cubes.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--cfg", type=str, default="cfg/sb3_sac.yaml")
    parser.add_argument("--logdir", type=str, default="logs/sb3/multi_order_cubes_sac")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .zip SB3 checkpoint to resume from.")
    parser.add_argument("--no_vecnormalize", action="store_true", default=False)
    parser.add_argument("--keep_all_info", action="store_true", default=False, help="Slower wrapper but keeps extra info.")
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_interval", type=int, default=20_000)
    parser.add_argument("--video_length", type=int, default=400)

    # IsaacLab app args (headless, device, enable_cameras, etc.)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DumpLoggerCallback(BaseCallback):
    """Force SB3 to dump logs to TensorBoard every `dump_freq` environment steps."""
    def __init__(self, dump_freq: int = 10_000):
        super().__init__()
        self.dump_freq = int(dump_freq)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.dump_freq == 0:
            # Writes whatever is currently recorded (including train/* metrics if available)
            self.model.logger.dump(self.num_timesteps)
        return True


class IsaacInfoTensorboardCallback(BaseCallback):
    """
    Lee infos[] y vuelca a TensorBoard cualquier key que empiece por 'moc/'.
    Recomendado entrenar con --keep_all_info para que Sb3VecEnvWrapper no filtre extras.
    """
    def __init__(self, log_every: int = 100):
        super().__init__()
        self.log_every = int(log_every)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every != 0:
            return True

        infos = self.locals.get("infos", None)
        if not infos:
            return True

        # infos is list length num_envs; we aggregate over envs
        # We collect moc/* tensors or scalars.
        agg = {}
        for info in infos:
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if not isinstance(k, str) or not k.startswith("moc/"):
                    continue
                agg.setdefault(k, []).append(v)

        if not agg:
            return True

        import numpy as np
        import torch

        for k, vals in agg.items():
            # vals: list of tensors / arrays / scalars (per-env)
            try:
                if isinstance(vals[0], torch.Tensor):
                    x = torch.stack([vv.float().mean() if vv.ndim > 0 else vv.float() for vv in vals]).mean().item()
                else:
                    # numpy/scalar
                    arr = np.array([float(np.mean(vv)) for vv in vals], dtype=np.float32)
                    x = float(arr.mean())
                self.logger.record(k, x)
            except Exception:
                continue

        return True


class SaveBestModelOnEpRewCallback(BaseCallback):
    """
    Guarda el mejor modelo según mean episode reward (sobre una ventana reciente),
    SIN crear un segundo entorno (prohibido). Requiere episodios (timeout).
    """
    def __init__(self, save_dir: Path, check_freq: int = 10_000, min_episodes: int = 5, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.save_dir = Path(save_dir)
        self.check_freq = int(check_freq)
        self.min_episodes = int(min_episodes)
        self.best_mean_ep_rew = -float("inf")

    def _on_step(self) -> bool:
        # Chequeo periódico por steps (en SB3: "timesteps" ya cuenta steps de VecEnv)
        if self.num_timesteps % self.check_freq != 0:
            return True

        # ep_info_buffer se llena cuando hay "done" con info["episode"]
        ep_buf = getattr(self.model, "ep_info_buffer", None)
        if ep_buf is None or len(ep_buf) < self.min_episodes:
            return True

        # Cada elemento suele tener {"r": ep_return, "l": ep_len, "t": time}
        mean_ep_rew = sum(e["r"] for e in ep_buf) / len(ep_buf)

        # Log explícito en TB
        self.logger.record("rollout/ep_rew_mean_window", float(mean_ep_rew))

        if mean_ep_rew > self.best_mean_ep_rew:
            self.best_mean_ep_rew = float(mean_ep_rew)

            best_path = self.save_dir / "best_sac.zip"
            self.model.save(str(best_path))

            # Si hay VecNormalize, guárdalo junto al best para poder reproducir
            env = self.model.get_env()
            try:
                from stable_baselines3.common.vec_env import VecNormalize
                if isinstance(env, VecNormalize):
                    env.save(str(self.save_dir / "best_vecnormalize.pkl"))
            except Exception:
                pass

            if self.verbose:
                print(f"[BEST] Saved new best model: mean_ep_rew={self.best_mean_ep_rew:.3f} -> {best_path}")

        return True


def main():
    args = parse_args()

    # If recording video, cameras must be enabled (offscreen ok headless)
    if args.video:
        args.enable_cameras = True

    # Launch SimulationApp FIRST
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import AFTER app is created
    import numpy as np
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import VecNormalize

    from multi_order_cubes.config.ur10_gripper import UR10LongSuctionMultiOrderCubesEnvCfg

    # ---- logging dirs
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{run_stamp}"
    log_root = Path(args.logdir) / run_name
    tb_dir = log_root / "tb"
    ckpt_dir = log_root / "checkpoints"
    vid_dir = log_root / "videos"
    log_root.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    # ---- env cfg
    env_cfg = UR10LongSuctionMultiOrderCubesEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = args.device
    # recomendación: headless para rendimiento (render ralentiza)
    # env_cfg.sim.enable_cameras depende de tu cfg; el flag enable_cameras del App ya activa cámaras.

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Optional: record video (must wrap BEFORE Sb3VecEnvWrapper, since Sb3 wrapper must be last) :contentReference[oaicite:1]{index=1}
    if args.video:
        env = RecordVideo(
            env,
            video_folder=str(vid_dir),
            step_trigger=lambda step: (step % int(args.video_interval)) == 0,
            video_length=int(args.video_length),
            name_prefix="train",
        )

    # SB3 VecEnv adapter (must be LAST wrapper) :contentReference[oaicite:2]{index=2}
    sb3_env = Sb3VecEnvWrapper(env, fast_variant=not args.keep_all_info)



    # Optional: normalize obs/reward (often helps for SAC with dense shaping)
    if not args.no_vecnormalize:
        sb3_env = VecNormalize(sb3_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ---- agent cfg
    # Resolve YAML path relative to this script if needed
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        script_dir = Path(__file__).resolve().parent
        cfg_path = (script_dir.parent / args.cfg).resolve()

    agent_cfg_raw = load_yaml(str(cfg_path))

    agent_cfg = process_sb3_cfg(agent_cfg_raw, num_envs=sb3_env.num_envs)  # official helper :contentReference[oaicite:3]{index=3}


    # ---- callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000, sb3_env.num_envs * 200),  # coarse default
        save_path=str(ckpt_dir),
        name_prefix="sac",
        save_replay_buffer=True,
        save_vecnormalize=not args.no_vecnormalize,
    )
    info_tb_cb = IsaacInfoTensorboardCallback(log_every=100)

    # ---- create / load model
    if args.checkpoint is not None:
        model = SAC.load(
            args.checkpoint,
            env=sb3_env,
            tensorboard_log=str(tb_dir),
            device="cuda" if "cuda" in str(args.device).lower() else "cpu",
            print_system_info=True,
        )
    else:
        model = SAC(
            env=sb3_env,
            tensorboard_log=str(tb_dir),
            device="cuda" if "cuda" in str(args.device).lower() else "cpu",
            seed=int(args.seed),
            **agent_cfg,
        )



    # ---- train
    dump_cb = DumpLoggerCallback(dump_freq=10_000)

    best_cb = SaveBestModelOnEpRewCallback(
        save_dir=log_root,
        check_freq=10_000,
        min_episodes=5,
        verbose=1,
    )

    obs, _ = env.reset()

    robot = env.scene["robot"]
    print("\n=== BODY NAMES ===")
    for i, name in enumerate(robot.data.body_names):
        print(i, name)
    print("==================\n")

    model.learn(
        total_timesteps=int(args.total_timesteps),
        callback=[checkpoint_cb, dump_cb, best_cb, info_tb_cb],
        log_interval=10,
        progress_bar=True,
    )

    # ---- save final
    model.save(str(log_root / "final_sac.zip"))
    if not args.no_vecnormalize and isinstance(sb3_env, VecNormalize):
        sb3_env.save(str(log_root / "vecnormalize.pkl"))

    # Close
    sb3_env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()