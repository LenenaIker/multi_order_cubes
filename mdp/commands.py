from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Prioritized from-slot sampling (2026-09-02): counters the "rich get richer" dynamic where
# consume_next_signal's success-gated resample lets an already-mastered slot cycle through many
# more episodes than a still-unsolved one, which just piles up dead time instead of ever
# practicing the hard slot. Weighting the draw by (1 - success_rate) biases new commands toward
# whichever slot is currently worst, without touching the "no free bail" gate itself.
_SLOT_SUCCESS_EMA_BETA = 0.05    # how fast the per-slot estimate reacts to new outcomes
_SLOT_PRIORITY_ALPHA = 1.0       # sharpness of the (1 - success_rate) weighting
_SLOT_PRIORITY_FLOOR = 0.05      # minimum relative weight for a solved slot, so it never stops being practiced


def _ensure_slot_success_ema(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if not hasattr(env, "moc_slot_success_ema") or env.moc_slot_success_ema is None:
        env.moc_slot_success_ema = torch.zeros((4,), dtype=torch.float32, device=env.device)
    return env.moc_slot_success_ema


def update_slot_success_ema(
    env: "ManagerBasedRLEnv",
    from_slot_0based: torch.Tensor,
    success_mask: torch.Tensor,
) -> None:
    """Update the global (not per-env) per-slot success EMA with a batch of command outcomes.

    `from_slot_0based`/`success_mask` are equal-length 1D tensors, one entry per env whose
    command just ended (either resampled after success, or reset while still unsolved). Looped
    over the 4 fixed slot values (cheap, num_slots is a hard constant) rather than scattered,
    so multiple envs reporting the same slot in one call average together instead of
    overwriting each other.
    """
    ema = _ensure_slot_success_ema(env)
    if from_slot_0based.numel() == 0:
        return

    outcome = success_mask.to(torch.float32)
    for s in range(4):
        mask = from_slot_0based == s
        if mask.any():
            batch_rate = outcome[mask].mean()
            ema[s] = (1.0 - _SLOT_SUCCESS_EMA_BETA) * ema[s] + _SLOT_SUCCESS_EMA_BETA * batch_rate
    ema.clamp_(0.0, 1.0)


def _ensure_command_buffers(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        env.command_from_to = torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)

    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        env.target_cube_id = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)


def _ensure_slot_mapping_buffers(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "moc_active_cube_slot_idx") or env.moc_active_cube_slot_idx is None:
        env.moc_active_cube_slot_idx = torch.zeros((env.num_envs, 3), dtype=torch.long, device=env.device)

    if not hasattr(env, "moc_slot_to_active_id") or env.moc_slot_to_active_id is None:
        env.moc_slot_to_active_id = -torch.ones((env.num_envs, 4), dtype=torch.long, device=env.device)


def latch_target_cube_from_command(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None = None,
) -> None:
    _ensure_command_buffers(env)
    _ensure_slot_mapping_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    cmd = env.command_from_to.index_select(0, env_ids)
    from_idx = torch.clamp(cmd[:, 0] - 1, 0, 3)

    slot_to_active = env.moc_slot_to_active_id.index_select(0, env_ids)
    row = torch.arange(env_ids.numel(), device=env.device)
    target_id = slot_to_active[row, from_idx]

    if (target_id < 0).any():
        bad_env_ids = env_ids[target_id < 0].detach().cpu().tolist()
        raise RuntimeError(f"[MOC] from_slot is empty for env_ids={bad_env_ids}")

    env.target_cube_id[env_ids] = target_id


def set_command_from_to(env: "ManagerBasedRLEnv", from_slot_1based: int, to_slot_1based: int) -> None:
    """Force the same (from, to) command on every env and latch the target cube for it.

    External entry point for an outside caller (e.g. a Cosmos Reason planner, or manual
    debugging) to hand the policy a specific command instead of letting it self-sample one.
    """
    _ensure_command_buffers(env)
    _ensure_slot_mapping_buffers(env)

    env.command_from_to[:, 0] = int(from_slot_1based)
    env.command_from_to[:, 1] = int(to_slot_1based)

    latch_target_cube_from_command(env)


def sample_command_from_to(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None = None,
    num_slots: int = 4,
) -> None:
    assert num_slots == 4, "This implementation assumes exactly 4 slots."

    _ensure_command_buffers(env)
    _ensure_slot_mapping_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
        if env_ids.numel() == 0:
            return

    slot_to_active = env.moc_slot_to_active_id.index_select(0, env_ids)

    occupied = slot_to_active >= 0
    empty = ~occupied

    success_ema = _ensure_slot_success_ema(env)
    priority = (1.0 - success_ema).clamp(min=0.0) ** _SLOT_PRIORITY_ALPHA + _SLOT_PRIORITY_FLOOR
    probs_from = occupied.to(torch.float32) * priority.view(1, 4)
    probs_from = probs_from / probs_from.sum(dim=1, keepdim=True).clamp(min=1e-6)
    from_idx = torch.multinomial(probs_from, num_samples=1).squeeze(1)

    probs_to = empty.to(torch.float32)
    sum_to = probs_to.sum(dim=1, keepdim=True)

    fallback_to = torch.ones_like(probs_to)
    fallback_to.scatter_(1, from_idx.view(-1, 1), 0.0)
    fallback_to = fallback_to / fallback_to.sum(dim=1, keepdim=True).clamp(min=1.0)

    probs_to = torch.where(sum_to > 0, probs_to / sum_to.clamp(min=1.0), fallback_to)
    to_idx = torch.multinomial(probs_to, num_samples=1).squeeze(1)

    env.command_from_to[env_ids, 0] = from_idx + 1
    env.command_from_to[env_ids, 1] = to_idx + 1

    latch_target_cube_from_command(env, env_ids)