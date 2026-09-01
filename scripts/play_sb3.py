import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
import play_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="Play SAC policy with SB3 on multi_order_cubes.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, default=None, help="Path to vecnormalize.pkl (optional).")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--diag_csv", type=str, default=None,
        help="Optional path to dump per-(step,env) diagnostic rows as CSV. Off by default; "
             "the stdout summary tables are always printed regardless of this flag.",
    )
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
    # Sb3VecEnvWrapper/VecNormalize store this exact object as self.env, and fast_variant=True
    # (below) discards all of env.extras before it ever reaches SB3's `infos` -- keep this direct
    # handle so the play loop can read env.extras itself, unaffected by that filtering.
    raw_env = env
    env = Sb3VecEnvWrapper(env, fast_variant=True)

    if args.vecnormalize is not None:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False

    model = SAC.load(args.checkpoint, env=env, device="cuda" if "cuda" in str(args.device).lower() else "cpu")

    obs = env.reset()
    rows: list[dict] = []
    # State right after the initial reset; paired with the reward computed by the FIRST
    # env.step() call below (see the re-snapshot comment inside the loop for why).
    snap = play_diagnostics.snapshot_cube_state(raw_env)
    prev_facts = None

    for i in range(int(args.steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        new_rows, prev_facts = play_diagnostics.build_rows(i, snap, raw_env.extras, rewards, dones, prev_facts)
        rows.extend(new_rows)

        # Re-snapshot AFTER step(): a reset or a mid-episode NEXT press can mutate
        # target_cube_id/active_cube_indices/moc_slot_to_active_id in place inside the step()
        # call just made (see mdp/events.py, mdp/commands.py). Snapshotting here pairs the NEW
        # target identity with the NEXT iteration's reward, instead of misattributing this
        # step's reward to whatever target/slot came after it.
        snap = play_diagnostics.snapshot_cube_state(raw_env)

        if i % 20 == 0:
            print(rewards)

    play_diagnostics.aggregate_and_print(rows)
    if args.diag_csv:
        play_diagnostics.write_csv(rows, Path(args.diag_csv))
        print(f"Wrote {len(rows)} rows to {args.diag_csv}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()