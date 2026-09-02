import argparse
from pathlib import Path
import datetime
import re
import yaml
import copy

from isaaclab.app import AppLauncher

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC with SB3 on multi_order_cubes.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--cfg", type=str, default="cfg/sb3_sac.yaml")
    parser.add_argument("--logdir", type=str, default="logs/sb3/multi_order_cubes_sac")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .zip SB3 checkpoint to resume from.")
    parser.add_argument(
        "--lr_start",
        type=float,
        default=None,
        help=(
            "Only used with --checkpoint. SAC.load() restores the schedule saved in the "
            "checkpoint as-is, ignoring --cfg's learning_rate, and reset_num_timesteps=True "
            "on model.learn() restarts that schedule's progress from 1.0 -- so a resumed run "
            "would otherwise jump back to the ORIGINAL run's peak LR. Pass --lr_start (and "
            "optionally --lr_end) to override with a fresh linear schedule for this run."
        ),
    )
    parser.add_argument(
        "--lr_end",
        type=float,
        default=0.0,
        help="Only used with --checkpoint and --lr_start. Value the linear schedule decays to.",
    )
    parser.add_argument(
        "--ent_coef_start",
        type=float,
        default=None,
        help=(
            "Only used with --checkpoint, and only if it was trained with ent_coef='auto'. "
            "SAC.load() restores the checkpoint's already-converged (and still shrinking, its "
            "Adam optimizer state carries over too) entropy coefficient as-is, so a resumed run "
            "inherits ever-less exploration noise across successive resumes instead of getting "
            "a real chance to explore into a new reward shape. Pass --ent_coef_start to reset "
            "log_ent_coef to log(this value) and give it a fresh Adam optimizer for this run."
        ),
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=None,
        help=(
            "Only used with --checkpoint. SAC.load() restores the checkpoint's own "
            "learning_starts (default 250_000, sized for a random-init policy), and since "
            "reset_num_timesteps=True resets num_timesteps to 0 on resume, a resumed run "
            "re-triggers that same random-action warmup window before any real training "
            "happens -- 250k steps of literal noise thrown at an already-competent policy, "
            "discarding it instead of using it. This is the wrong regime for a warm resume: "
            "the published fix (WSRL, 'Efficient Online RL Fine-Tuning Need Not Retain "
            "Offline Data', ICLR 2025) is to seed the resumed buffer with rollouts FROM the "
            "already-trained policy, not random actions, which is exactly what a small/zero "
            "learning_starts gets you here since SAC only samples randomly while "
            "num_timesteps < learning_starts. Pass e.g. --learning_starts 0 (or a few "
            "thousand, if you want a brief buffer-refill margin) on a resumed run."
        ),
    )
    parser.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help=(
            "Only used with --checkpoint. Path to a VecNormalize .pkl to restore obs "
            "normalization stats from, matching the checkpoint's own training run. If "
            "omitted, inferred from --checkpoint by this script's own save-file naming "
            "convention (final_sac.zip -> vecnormalize.pkl, best_sac.zip -> "
            "best_vecnormalize.pkl, checkpoints/<prefix>_<N>_steps.zip -> "
            "checkpoints/<prefix>_vecnormalize_<N>_steps.pkl, sibling to the checkpoint). "
            "Without a match, VecNormalize starts with cold (empty) running statistics "
            "even though the loaded policy was trained against the old run's converged "
            "statistics -- a real distribution-shift tax on top of the replay buffer's own "
            "cold start (see --checkpoint's own docstring caveat)."
        ),
    )
    parser.add_argument("--no_vecnormalize", action="store_true", default=False)
    parser.add_argument("--keep_all_info", action="store_true", default=False, help="Slower wrapper but keeps extra info.")
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_interval", type=int, default=20_000)
    parser.add_argument("--video_length", type=int, default=400)

    
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def infer_vecnormalize_path(checkpoint_path: Path) -> Path | None:
    """Infers the sibling VecNormalize .pkl saved alongside a checkpoint by this script's
    own naming convention (see CheckpointCallback's save_vecnormalize / SaveBestModelOnEpRewCallback
    / the final sb3_env.save() call below), so a --checkpoint resume can restore obs
    normalization stats matching the checkpoint instead of always starting VecNormalize cold.
    Returns None if the checkpoint filename doesn't match any known pattern, or the inferred
    file doesn't actually exist -- callers should treat both the same (fall back / warn).
    """
    name = checkpoint_path.name
    parent = checkpoint_path.parent
    if name == "final_sac.zip":
        candidate = parent / "vecnormalize.pkl"
    elif name == "best_sac.zip":
        candidate = parent / "best_vecnormalize.pkl"
    else:
        m = re.match(r"^(.+)_(\d+)_steps\.zip$", name)
        if not m:
            return None
        prefix, steps = m.group(1), m.group(2)
        candidate = parent / f"{prefix}_vecnormalize_{steps}_steps.pkl"
    return candidate if candidate.is_file() else None


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
            
            self.model.logger.dump(self.num_timesteps)
        return True


class IsaacInfoTensorboardCallback(BaseCallback):
    """
    Lee infos[] y vuelca a TensorBoard cualquier key cuyo prefijo esté en LOGGED_PREFIXES.

    Recomendado entrenar con --keep_all_info para que Sb3VecEnvWrapper no filtre extras.

    Categories (cubes/, position/, grip/, tasks/) are each their own top-level tag prefix,
    deliberately not nested under a shared 'moc/' umbrella: TensorBoard's scalar dashboard
    only groups cards by the substring before the FIRST '/' in a tag name, it doesn't
    recurse into further slashes -- confirmed empirically (15 moc/* tags all landing flat
    under one 'moc' group, overflowing its 12-card page limit). Making the category itself
    the first segment gives each one its own top-level, separately-paginated group instead
    of one giant flat bucket. Trade-off: these are generic English words with no 'moc'
    marker, so a future unrelated env.extras key starting with the same word would also get
    swept into TensorBoard by this filter -- acceptable here since env.extras is only ever
    populated by this project's own mdp/*.py reward and termination functions.
    """

    LOGGED_PREFIXES = ("cubes/", "position/", "grip/", "tasks/", "rewards/")

    def __init__(self, log_every: int = 100):
        super().__init__()
        self.log_every = int(log_every)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every != 0:
            return True

        infos = self.locals.get("infos", None)
        if not infos:
            return True



        agg = {}
        for info in infos:
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if not isinstance(k, str) or not k.startswith(self.LOGGED_PREFIXES):
                    continue
                agg.setdefault(k, []).append(v)

        if not agg:
            return True

        import numpy as np
        import torch

        for k, vals in agg.items():
            
            try:
                if isinstance(vals[0], torch.Tensor):
                    x = torch.stack([vv.float().mean() if vv.ndim > 0 else vv.float() for vv in vals]).mean().item()
                else:
                    
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
        
        if self.num_timesteps % self.check_freq != 0:
            return True

        
        ep_buf = getattr(self.model, "ep_info_buffer", None)
        if ep_buf is None or len(ep_buf) < self.min_episodes:
            return True

        
        mean_ep_rew = sum(e["r"] for e in ep_buf) / len(ep_buf)

        
        self.logger.record("rollout/ep_rew_mean_window", float(mean_ep_rew))

        if mean_ep_rew > self.best_mean_ep_rew:
            self.best_mean_ep_rew = float(mean_ep_rew)

            best_path = self.save_dir / "best_sac.zip"
            self.model.save(str(best_path))

            
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

    
    if args.video:
        args.enable_cameras = True

    
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    
    import numpy as np
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import VecNormalize

    from multi_order_cubes.config.ur10_gripper import UR10LongSuctionMultiOrderCubesEnvCfg

    
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

    
    env_cfg = UR10LongSuctionMultiOrderCubesEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = args.device
    
    

    env = ManagerBasedRLEnv(cfg=env_cfg)

    
    if args.video:
        env = RecordVideo(
            env,
            video_folder=str(vid_dir),
            step_trigger=lambda step: (step % int(args.video_interval)) == 0,
            video_length=int(args.video_length),
            name_prefix="train",
        )

    
    sb3_env = Sb3VecEnvWrapper(env, fast_variant=not args.keep_all_info)

    
    if not args.no_vecnormalize:
        vecnorm_path = None
        if args.checkpoint is not None:
            vecnorm_path = Path(args.vecnormalize) if args.vecnormalize else infer_vecnormalize_path(Path(args.checkpoint))
            if vecnorm_path is not None and not vecnorm_path.is_file():
                print(f"[WARN] --vecnormalize path not found: {vecnorm_path} -- starting VecNormalize cold.")
                vecnorm_path = None
            elif vecnorm_path is None:
                print(
                    "[WARN] Resuming from --checkpoint but no matching VecNormalize .pkl was found "
                    "(pass --vecnormalize explicitly if it exists under a different name) -- "
                    "starting VecNormalize cold, mismatched against the loaded policy's training scale."
                )

        if vecnorm_path is not None:
            sb3_env = VecNormalize.load(str(vecnorm_path), sb3_env)
            sb3_env.training = True
            sb3_env.norm_reward = False
            print(f"[INFO] Restored VecNormalize stats from {vecnorm_path}")
        else:
            sb3_env = VecNormalize(sb3_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    
    
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        script_dir = Path(__file__).resolve().parent
        cfg_path = (script_dir.parent / args.cfg).resolve()

    agent_cfg_raw = load_yaml(str(cfg_path))

    agent_cfg = process_sb3_cfg(agent_cfg_raw, num_envs=sb3_env.num_envs)

    # gradient_steps in cfg/sb3_sac.yaml is tuned against a num_envs=64 baseline: with
    # train_freq=1, SAC does `gradient_steps` updates per env.step() call regardless of
    # num_envs, so more envs means fewer calls for the same total_timesteps, and thus
    # proportionally fewer gradient updates unless we scale gradient_steps to compensate.
    # This keeps total gradient updates over the whole run roughly constant across
    # num_envs choices, so runs stay comparable and only the (parallel) env-stepping
    # speeds up with more envs, not the amount of actual training that happens.
    if "gradient_steps" in agent_cfg:
        _baseline_num_envs = 64
        scale = max(1, sb3_env.num_envs) / _baseline_num_envs
        agent_cfg["gradient_steps"] = max(1, round(agent_cfg["gradient_steps"] * scale))


    # Desired checkpoint interval in REAL env timesteps (across all parallel envs).
    checkpoint_every_timesteps = max(10_000, sb3_env.num_envs * 200)
    # CheckpointCallback counts *calls* to _on_step (one per vectorized env.step(),
    # i.e. num_envs real timesteps each), not real timesteps directly, so we convert.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, checkpoint_every_timesteps // sb3_env.num_envs),
        save_path=str(ckpt_dir),
        name_prefix="sac",
        # Replay buffer is preallocated to buffer_size (2M transitions), so each dump is
        # ~1GB flat regardless of how full it actually is, and CheckpointCallback never
        # deletes old ones. With only ~69GB free on disk, saving it every 12.8k steps
        # would fill the disk and kill a long unattended run well before it converges.
        save_replay_buffer=False,
        save_vecnormalize=not args.no_vecnormalize,
    )
    info_tb_cb = IsaacInfoTensorboardCallback(log_every=100)

    
    if args.checkpoint is not None:
        model = SAC.load(
            args.checkpoint,
            env=sb3_env,
            tensorboard_log=str(tb_dir),
            device="cuda" if "cuda" in str(args.device).lower() else "cpu",
            print_system_info=True,
        )

        if "gradient_steps" in agent_cfg:
            # SAC.load() restores gradient_steps as pickled in the checkpoint -- i.e. scaled
            # for whatever num_envs that checkpoint was ORIGINALLY trained at, not this run's
            # --num_envs. Left alone, a resume at a different num_envs silently keeps the old
            # gradient-updates-per-env.step() count, so the ratio of gradient updates to newly
            # collected transitions drifts away from what the scaling above intends (e.g.
            # resuming a 1024-env checkpoint at 4096 envs quietly trains at 1/4 the intended
            # update intensity). Re-apply the freshly computed, num_envs-scaled value here.
            model.gradient_steps = agent_cfg["gradient_steps"]

        if args.lr_start is not None:
            # SAC.load() restored whatever schedule was pickled in the checkpoint, and
            # model.learn() below resets num_timesteps to 0 (reset_num_timesteps=True is
            # SB3's default and we don't override it), which restarts that schedule's
            # progress_remaining from 1.0. Left alone, this makes a resumed run jump back to
            # the ORIGINAL run's peak learning_rate instead of continuing its decay. Build a
            # fresh linear schedule for *this* run instead.
            from stable_baselines3.common.utils import get_linear_fn

            model.learning_rate = get_linear_fn(float(args.lr_start), float(args.lr_end), end_fraction=1.0)
            # SB3 re-derives every optimizer's actual lr from model.lr_schedule at the start
            # of each train() call, so re-pointing it here is enough -- no need to touch the
            # actor/critic/ent_coef optimizers' param_groups directly.
            model._setup_lr_schedule()

        if args.ent_coef_start is not None:
            if model.log_ent_coef is None:
                print(
                    "--ent_coef_start given but this checkpoint was trained with a fixed "
                    "ent_coef (not 'auto') -- there is no log_ent_coef to reset, ignoring."
                )
            else:
                import math
                import torch

                with torch.no_grad():
                    model.log_ent_coef.fill_(math.log(float(args.ent_coef_start)))
                # The restored ent_coef_optimizer's Adam momentum was actively driving
                # log_ent_coef DOWN across every previous resume (confirmed: 0.0088 -> 0.0034
                # -> 0.0027 over three successive runs off the same lineage) -- a fresh
                # optimizer drops that inherited downward pressure so this run gets an honest
                # read on whether more exploration actually helps find grasp/lift.
                ent_lr = float(args.lr_start) if args.lr_start is not None else model.lr_schedule(1)
                model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=ent_lr)

        if args.learning_starts is not None:
            # See --learning_starts' own help text: SAC only samples random actions while
            # num_timesteps < learning_starts, and reset_num_timesteps=True means a resumed
            # run starts that count over from 0 again. Left at the checkpoint's original
            # value (sized for a random-init policy), this throws away an already-competent
            # actor for that whole window instead of using it to collect on-policy data.
            model.learning_starts = int(args.learning_starts)
            print(f"[INFO] Overrode learning_starts to {model.learning_starts} for this resumed run.")
    else:
        model = SAC(
            env=sb3_env,
            tensorboard_log=str(tb_dir),
            device="cuda" if "cuda" in str(args.device).lower() else "cpu",
            seed=int(args.seed),
            **agent_cfg,
        )



    
    dump_cb = DumpLoggerCallback(dump_freq=10_000)

    best_cb = SaveBestModelOnEpRewCallback(
        save_dir=log_root,
        check_freq=10_000,
        min_episodes=5,
        verbose=1,
    )

    model.learn(
        total_timesteps=int(args.total_timesteps),
        callback = [checkpoint_cb, info_tb_cb, dump_cb, best_cb],
        log_interval=10,
        progress_bar=True,
    )

    
    model.save(str(log_root / "final_sac.zip"))
    if not args.no_vecnormalize and isinstance(sb3_env, VecNormalize):
        sb3_env.save(str(log_root / "vecnormalize.pkl"))

    sb3_env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()