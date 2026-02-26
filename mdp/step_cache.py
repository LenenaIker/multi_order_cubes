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

    # Token used to decide whether cache is still valid for the *current* sim step.
    if not hasattr(env, "_moc_cache_token") or env._moc_cache_token is None:
        env._moc_cache_token = -1

    # Keep compatibility: some code increments this (NextFlagAction).
    if not hasattr(env, "_moc_step_id") or env._moc_step_id is None:
        env._moc_step_id = -1

        # Monotonic reset id (used to invalidate cache across resets even without env.step()).
    if not hasattr(env, "_moc_reset_id") or env._moc_reset_id is None:
        env._moc_reset_id = 0


def _get_step_token(env: "ManagerBasedRLEnv") -> int:
    """Best-effort token that changes every sim step.

    Priority:
      1) env._moc_step_id if user increments it reliably.
      2) Fallback to a reduced signature of episode_length_buf (+ reset_buf if present).

    This avoids stale caches even when NextFlagAction is not executed.
    """
    # Preferred: explicit monotonic step id
    try:
        sid = int(env._moc_step_id)
        if sid >= 0:
            return sid
    except Exception:
        pass

    # Fallback: use reductions over per-env step buffers (cheap and usually step-unique).
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        # Worst case fallback: always invalidate (token changes every call).
        # But we still return something deterministic-ish.
        return 0

    el = env.episode_length_buf
    s = int(el.sum().item())
    mx = int(el.max().item())
    mn = int(el.min().item())

    rb = 0
    if hasattr(env, "reset_buf") and env.reset_buf is not None:
        rb = int(env.reset_buf.sum().item())

    rid = 0
    if hasattr(env, "_moc_reset_id") and env._moc_reset_id is not None:
        rid = int(env._moc_reset_id)

    base = (s << 32) ^ (mx << 16) ^ (mn << 8) ^ rb
    return (rid << 48) ^ base

def _invalidate_if_needed(env: "ManagerBasedRLEnv") -> None:
    """Invalidate cache if env step changed."""
    _ensure_cache(env)
    token = _get_step_token(env)
    if env._moc_cache_token != token:
        env._moc_cache.clear()
        env._moc_cache_token = token


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

    # During env initialization, ObservationManager calls observation functions to infer shapes
    # before reset events run. In that phase, active_cube_indices may not exist yet.
    # Return a correctly-shaped zero tensor instead of crashing.
    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        N = int(getattr(env, "num_envs", 1))
        device = getattr(env, "device", "cpu")
        return torch.zeros((N, 3, 3), dtype=torch.float32, device=device)
    

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

    # During env initialization, ObservationManager calls observation functions to infer shapes
    # before reset events run. In that phase, active_cube_indices may not exist yet.
    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        N = int(getattr(env, "num_envs", 1))
        device = getattr(env, "device", "cpu")
        # identity quaternion (w,x,y,z) convention in IsaacLab is typically (w, x, y, z)
        q = torch.zeros((N, 3, 4), dtype=torch.float32, device=device)
        q[..., 0] = 1.0
        return q

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