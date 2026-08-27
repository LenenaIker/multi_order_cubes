from __future__ import annotations

import torch

from .step_cache import get_active_cube_pos_w


def time_out(env) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        raise AttributeError("env.episode_length_buf not found.")
    if not hasattr(env, "max_episode_length"):
        raise AttributeError("env.max_episode_length not found.")
    return env.episode_length_buf >= (int(env.max_episode_length) - 1)


def cube_fell_off_table(env, height_threshold: float = -0.05, max_xy_displacement: float = 1.0) -> torch.Tensor:
    """Also catches a cube flung sideways/upward by a physics glitch (elbow bump, contact
    blow-up), not just one that fell below the table. Height alone misses that case, and
    without this the episode runs to timeout with the cube's position stuck far outside the
    scene, feeding `penalty_bystander_displacement` an ever-growing, unbounded term every
    step for the rest of the episode (see moc/reward_wiring_status for the reward-divergence
    incident this caused).
    """
    cubes_pos = get_active_cube_pos_w(env)
    fell = cubes_pos[:, :, 2] < float(height_threshold)

    if not hasattr(env, "moc_cube_home_pos_w") or env.moc_cube_home_pos_w is None:
        result = fell.any(dim=1)
        if not hasattr(env, "extras") or env.extras is None:
            env.extras = {}
        env.extras["moc/cube_off_table"] = result
        return result

    xy_disp = torch.linalg.vector_norm(cubes_pos[:, :, :2] - env.moc_cube_home_pos_w[:, :, :2], dim=-1)
    flung = xy_disp > float(max_xy_displacement)

    result = (fell | flung).any(dim=1)

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/cube_off_table"] = result

    return result