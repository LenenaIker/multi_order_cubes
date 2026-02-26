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
from multi_order_cubes.mdp.terminations import move_success
from multi_order_cubes.mdp.rewards import reward_penalty_disturb_other_cubes
from multi_order_cubes.smoke_tests import (
    run_random_policy_smoke,
    run_adversarial_next_spam_smoke,
    run_disturb_push_smoke,
    run_no_false_positive_adjacent_slots_smoke,
    run_require_settled_no_crash_smoke,
    run_step_cache_invalidation_without_next_smoke,
    run_move_success_reset_diagnostics,
    run_no_start_in_success_smoke
)


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
        actions = torch.randn((env.num_envs, env.action_manager.total_action_dim), device=env.device) * 0.1

        obs, rew, terminated, truncated, info = env.step(actions)

        if t % 10 == 0:
            ok = move_success(env)
            disturb = reward_penalty_disturb_other_cubes(env)
            print(
                f"t={t} cmd={env.command_from_to[:].tolist()} "
                f"ok={ok[:].tolist()} disturb={disturb[:].tolist()} rew={rew[:].tolist()}"
            )

        if bool(terminated.any() or truncated.any()):
            obs, info = env.reset()
            set_command_from_to(env, 1, 2)
            ensure_moc_buffers(env)

    env.close()


def diagnostic_all_smoke():
    env_cfg = UR10LongSuctionMultiOrderCubesEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        # ----------------------------
        # FAST / STRUCTURAL SMOKES FIRST
        # ----------------------------
        run_no_start_in_success_smoke(env)                      # your key regression
        run_step_cache_invalidation_without_next_smoke(env)     # cache correctness without NEXT
        run_require_settled_no_crash_smoke(env)                 # settled gating must not crash
        run_no_false_positive_adjacent_slots_smoke(env)         # tol_xy / adjacency false positives
        run_move_success_reset_diagnostics(env)                 # prints reset diagnostics

        # ----------------------------
        # BEHAVIORAL / LONGER SMOKES
        # ----------------------------
        run_random_policy_smoke(env)                            # long random rollouts + metrics
        run_adversarial_next_spam_smoke(env)                    # spam NEXT
        run_disturb_push_smoke(env)                             # induce contact/disturb

    finally:
        env.close()


if __name__ == "__main__":
    diagnostic_all_smoke()
    simulation_app.close()
