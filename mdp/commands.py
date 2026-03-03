# mdp/commands.py
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from typing import TYPE_CHECKING

from .terminations import move_success
from .step_cache import (
    get_slots_w as _get_slots_w,
    get_active_cube_pos_w as _get_active_cube_pos_w,
    get_nearest_slot_for_active_cubes_xy as _get_nearest_slot_for_active_cubes_xy,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Cached scene geometry accessors (single source of truth = step_cache.py)
# -----------------------------------------------------------------------------

def slots_w(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """(N,4,3) slot positions in world frame (optionally indexed by env_ids)."""
    slots = _get_slots_w(env)
    if env_ids is None:
        return slots
    env_ids = env_ids.to(device=env.device)
    return slots.index_select(0, env_ids)


def active_cube_positions_w(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """(N,3,3) active cube positions in world frame (optionally indexed by env_ids)."""
    pos = _get_active_cube_pos_w(env)
    if env_ids is None:
        return pos
    env_ids = env_ids.to(device=env.device)
    return pos.index_select(0, env_ids)


def nearest_slot_for_active_cubes_xy(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """(N,3) nearest slot index (0..3) for each active cube, optionally indexed by env_ids."""
    nearest = _get_nearest_slot_for_active_cubes_xy(env)
    if env_ids is None:
        return nearest
    env_ids = env_ids.to(device=env.device)
    return nearest.index_select(0, env_ids)


# -----------------------------------------------------------------------------
# Buffer initialization (split by responsibility)
# -----------------------------------------------------------------------------

def ensure_command_buffer(env: ManagerBasedRLEnv):
    """Create env.command_from_to if it doesn't exist."""
    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        env.command_from_to = torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)


def ensure_command_buffers(env: ManagerBasedRLEnv) -> None:
    """Buffers related to the discrete (from,to) command and its latched context."""
    ensure_command_buffer(env)
    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        env.target_cube_id = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    if not hasattr(env, "moc_cmd_cube_pos_xy0") or env.moc_cmd_cube_pos_xy0 is None:
        env.moc_cmd_cube_pos_xy0 = torch.zeros((env.num_envs, 3, 2), dtype=torch.float32, device=env.device)
    if not hasattr(env, "moc_cmd_stamp") or env.moc_cmd_stamp is None:
        env.moc_cmd_stamp = -torch.ones((env.num_envs,), dtype=torch.long, device=env.device)


# def ensure_phase_buffers(env: ManagerBasedRLEnv) -> None:
#     """Buffers for the phase machine."""
#     if not hasattr(env, "moc_phase") or env.moc_phase is None:
#         env.moc_phase = torch.ones((env.num_envs,), dtype=torch.int32, device=env.device)
#     if not hasattr(env, "moc_phase_hold") or env.moc_phase_hold is None:
#         env.moc_phase_hold = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)
#     if not hasattr(env, "moc_prev_ep_len") or env.moc_prev_ep_len is None:
#         env.moc_prev_ep_len = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)


def ensure_next_buffers_light(env: ManagerBasedRLEnv) -> None:
    """Buffers for NEXT edge detection (kept minimal)."""
    if not hasattr(env, "moc_next_prev") or env.moc_next_prev is None:
        env.moc_next_prev = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)


# --- REPLACE in mdp/commands.py ---
def ensure_moc_buffers(env: "ManagerBasedRLEnv") -> None:
    """Minimal buffers needed by MOC (no phase machine)."""
    ensure_command_buffers(env)
    ensure_next_buffers_light(env)


# -----------------------------------------------------------------------------
# Command latching / sampling
# -----------------------------------------------------------------------------

def latch_command_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """Latch per-command auxiliary state without resampling the command."""
    ensure_command_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
        if env_ids.numel() == 0:
            return

    cmd = env.command_from_to.index_select(0, env_ids)
    from_idx = torch.clamp(cmd[:, 0] - 1, 0, 3)  # (M,)

    cubes_pos = active_cube_positions_w(env, env_ids=env_ids)  # (M,3,3)
    slots = slots_w(env, env_ids=env_ids)                      # (M,4,3)

    nearest = nearest_slot_for_active_cubes_xy(env).index_select(0, env_ids)  # (M,3)
    match = (nearest == from_idx.unsqueeze(1))                                # (M,3)
    has_match = match.any(dim=1)                                              # (M,)
    first_match_id = match.to(torch.int64).argmax(dim=1)                      # (M,)

    from_slot_xy = slots[torch.arange(env_ids.numel(), device=env.device), from_idx, :2]  # (M,2)
    dxy = cubes_pos[:, :, :2] - from_slot_xy.unsqueeze(1)                                 # (M,3,2)
    dist2_from = (dxy * dxy).sum(dim=-1)                                                   # (M,3)
    fallback_id = dist2_from.argmin(dim=1)                                                 # (M,)

    target_id = torch.where(has_match, first_match_id, fallback_id)                        # (M,)

    env.target_cube_id[env_ids] = target_id
    env.moc_cmd_cube_pos_xy0[env_ids] = cubes_pos[:, :, :2]
    if hasattr(env, "episode_length_buf"):
        env.moc_cmd_stamp[env_ids] = env.episode_length_buf[env_ids]


def set_command_from_to(env: ManagerBasedRLEnv, from_slot_1based: int, to_slot_1based: int):
    """Set same command for all envs (debug) AND latch target/baselines."""
    ensure_command_buffers(env)
    env.command_from_to[:, 0] = int(from_slot_1based)
    env.command_from_to[:, 1] = int(to_slot_1based)
    latch_command_state(env)


def sample_command_from_to(
    env: ManagerBasedRLEnv,
    num_slots: int = 4,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized sampling of (from,to) per-env, supports partial update with env_ids."""
    assert num_slots == 4, "This implementation assumes 4 slots"
    ensure_command_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
        if env_ids.numel() == 0:
            return env.command_from_to

    nearest_all = nearest_slot_for_active_cubes_xy(env)          # (N,3)
    nearest = nearest_all.index_select(0, env_ids)               # (M,3)
    M = int(env_ids.numel())

    # occupancy multi-hot (M,4)
    occ = torch.zeros((M, num_slots), dtype=torch.bool, device=env.device)
    occ.scatter_(1, nearest, True)
    empty = ~occ

    # sample FROM among occupied
    probs_from = occ.to(torch.float32)
    sum_from = probs_from.sum(dim=1, keepdim=True)
    fallback_from = torch.full_like(probs_from, 1.0 / float(num_slots))
    probs_from = torch.where(sum_from > 0, probs_from / sum_from.clamp(min=1.0), fallback_from)
    from_idx = torch.multinomial(probs_from, num_samples=1).squeeze(1)  # 0..3

    # sample TO among empty
    probs_to = empty.to(torch.float32)
    sum_to = probs_to.sum(dim=1, keepdim=True)
    fallback_to = torch.ones_like(probs_to)
    fallback_to.scatter_(1, from_idx.view(-1, 1), 0.0)  # avoid to==from
    fallback_to = fallback_to / fallback_to.sum(dim=1, keepdim=True).clamp(min=1.0)
    probs_to = torch.where(sum_to > 0, probs_to / sum_to.clamp(min=1.0), fallback_to)
    to_idx = torch.multinomial(probs_to, num_samples=1).squeeze(1)  # 0..3

    # store 1-based
    env.command_from_to[env_ids, 0] = from_idx + 1
    env.command_from_to[env_ids, 1] = to_idx + 1

    # latch target/baseline for these env_ids
    latch_command_state(env, env_ids=env_ids)
    return env.command_from_to