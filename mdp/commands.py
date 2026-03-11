from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

    probs_from = occupied.to(torch.float32)
    probs_from = probs_from / probs_from.sum(dim=1, keepdim=True).clamp(min=1.0)
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