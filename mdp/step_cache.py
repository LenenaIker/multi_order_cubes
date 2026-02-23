from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CUBE_KEYS_9 = [
    "cube_light_s", "cube_light_m", "cube_light_l",
    "cube_flat_s",  "cube_flat_m",  "cube_flat_l",
    "cube_dark_s",  "cube_dark_m",  "cube_dark_l",
]


def _ensure_cache(env: "ManagerBasedRLEnv") -> None:
    """Initialize cache containers lazily."""
    if not hasattr(env, "_moc_cache") or env._moc_cache is None:
        env._moc_cache = {}
    if not hasattr(env, "_moc_cache_step_id") or env._moc_cache_step_id is None:
        env._moc_cache_step_id = -1

    # If step_id doesn't exist (e.g., if NextFlagAction not used), fall back to -1.
    if not hasattr(env, "_moc_step_id") or env._moc_step_id is None:
        env._moc_step_id = -1


def _invalidate_if_needed(env: "ManagerBasedRLEnv") -> None:
    """Invalidate cache if env step changed."""
    _ensure_cache(env)
    sid = int(env._moc_step_id)
    if env._moc_cache_step_id != sid:
        env._moc_cache.clear()
        env._moc_cache_step_id = sid


def get_slots_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,4,3) slot positions in world frame."""
    _invalidate_if_needed(env)
    key = "slots_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    slots_local = torch.as_tensor(env.cfg.slot_positions, dtype=torch.float32, device=env.device)  # (4,3)
    origins = env.scene.env_origins  # (N,3)
    slots_w = slots_local.unsqueeze(0) + origins.unsqueeze(1)  # (N,4,3)

    env._moc_cache[key] = slots_w
    return slots_w


def get_cube_pos9_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,9,3) positions for all 9 cube assets."""
    _invalidate_if_needed(env)
    key = "pos9_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    pos9 = torch.stack([env.scene[k].data.root_pos_w for k in CUBE_KEYS_9], dim=1)  # (N,9,3)
    env._moc_cache[key] = pos9
    return pos9


def get_cube_quat9_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,9,4) quaternions for all 9 cube assets."""
    _invalidate_if_needed(env)
    key = "quat9_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    quat9 = torch.stack([env.scene[k].data.root_quat_w for k in CUBE_KEYS_9], dim=1)  # (N,9,4)
    env._moc_cache[key] = quat9
    return quat9


def get_active_cube_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,3,3) positions for active cubes only."""
    _invalidate_if_needed(env)
    key = "active_pos_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        raise RuntimeError("env.active_cube_indices missing. Ensure reset event randomize_cubes_on_slots ran.")

    pos9 = get_cube_pos9_w(env)  # (N,9,3)
    idx = env.active_cube_indices  # (N,3)
    active = pos9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (N,3,3)

    env._moc_cache[key] = active
    return active


def get_active_cube_quat_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,3,4) quaternions for active cubes only."""
    _invalidate_if_needed(env)
    key = "active_quat_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        raise RuntimeError("env.active_cube_indices missing. Ensure reset event randomize_cubes_on_slots ran.")

    quat9 = get_cube_quat9_w(env)  # (N,9,4)
    idx = env.active_cube_indices  # (N,3)
    active = quat9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))  # (N,3,4)

    env._moc_cache[key] = active
    return active


def get_nearest_slot_for_active_cubes_xy(env: "ManagerBasedRLEnv", num_slots: int = 4) -> torch.Tensor:
    """(N,3) nearest slot index per active cube using XY distance."""
    _invalidate_if_needed(env)
    key = "nearest_slot_active_xy"
    if key in env._moc_cache:
        return env._moc_cache[key]

    cubes = get_active_cube_pos_w(env)[:, :, :2]  # (N,3,2)
    slots = get_slots_w(env)[:, :, :2]           # (N,4,2)

    d = cubes.unsqueeze(2) - slots.unsqueeze(1)  # (N,3,4,2)
    dist2 = (d * d).sum(dim=-1)                  # (N,3,4)
    nearest = torch.argmin(dist2, dim=2)          # (N,3)

    env._moc_cache[key] = nearest
    return nearest