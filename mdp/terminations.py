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

    The target cube is excluded from this check (mirrors `penalty_bystander_displacement`'s
    own `is_target` masking). A 20M-step run showed the agent had learned to hover at the
    exact reward-maximizing distance with the gripper closed, never attempting real contact
    -- ending the whole episode over a fall/fling of the one cube it's actually supposed to
    touch was a real, rational deterrent, not a hidden penalty. Bystanders still terminate
    the episode on fall/fling as before; a knocked-off target cube just lets the episode run
    on, naturally losing the (much larger) reach reward for the rest of it instead of a hard
    cliff.
    """
    cubes_pos = get_active_cube_pos_w(env)
    fell = cubes_pos[:, :, 2] < float(height_threshold)

    target_id = env.target_cube_id.to(torch.long).clamp(0, fell.shape[1] - 1)
    is_target = torch.zeros_like(fell)
    is_target.scatter_(1, target_id.view(-1, 1), True)
    fell = fell & ~is_target

    if not hasattr(env, "moc_cube_home_pos_w") or env.moc_cube_home_pos_w is None:
        result = fell.any(dim=1)
        if not hasattr(env, "extras") or env.extras is None:
            env.extras = {}
        env.extras["moc/cube_off_table"] = result
        return result

    xy_disp = torch.linalg.vector_norm(cubes_pos[:, :, :2] - env.moc_cube_home_pos_w[:, :, :2], dim=-1)
    flung = (xy_disp > float(max_xy_displacement)) & ~is_target

    result = (fell | flung).any(dim=1)

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/cube_off_table"] = result

    return result