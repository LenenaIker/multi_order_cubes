# mdp/commands.py
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _slots_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(4,3) slots in world frame as torch on env.device."""
    slots = torch.as_tensor(env.cfg.slot_positions, dtype=torch.float32, device=env.device)
    assert slots.shape == (4, 3), f"Expected slot_positions shape (4,3), got {slots.shape}"
    return slots


CUBE_KEYS_9 = [
    "cube_light_s", "cube_light_m", "cube_light_l",
    "cube_flat_s",  "cube_flat_m",  "cube_flat_l",
    "cube_dark_s",  "cube_dark_m",  "cube_dark_l",
]

def _active_cube_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns (N,3,3) positions for the 3 ACTIVE cubes (one per color), per-env.
    Requires env.active_cube_indices set by reset event.
    """
    assert hasattr(env, "active_cube_indices"), "env.active_cube_indices missing. Call reset event randomize_cubes_on_slots first."

    # (N,9,3)
    pos9 = torch.stack([env.scene[k].data.root_pos_w for k in CUBE_KEYS_9], dim=1)

    idx = env.active_cube_indices  # (N,3) values 0..8
    idx3 = idx.unsqueeze(-1).expand(-1, -1, 3)  # (N,3,3)

    return pos9.gather(1, idx3)

# Replace old uses:
# _cube_positions_w(...) -> _active_cube_positions_w(env)
def _nearest_slot_for_each_cube_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    cubes = _active_cube_positions_w(env)[:, :, :2]  # (N,3,2)
    slots = _slots_w(env)[:, :2].unsqueeze(0)        # (1,4,2)
    d = cubes.unsqueeze(2) - slots.unsqueeze(1)      # (N,3,4,2)
    dist2 = (d * d).sum(dim=-1)                      # (N,3,4)
    return torch.argmin(dist2, dim=2)                # (N,3)


def ensure_command_buffer(env: ManagerBasedRLEnv):
    """Create env.command_from_to if it doesn't exist."""
    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        env.command_from_to = torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)


def set_command_from_to(env: ManagerBasedRLEnv, from_slot_1based: int, to_slot_1based: int):
    """Set same command for all envs (useful for debugging)."""
    ensure_command_buffer(env)
    env.command_from_to[:, 0] = int(from_slot_1based)
    env.command_from_to[:, 1] = int(to_slot_1based)

def sample_command_from_to(
    env: ManagerBasedRLEnv,
    num_slots: int = 4,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Vectorized sampling of (from,to) per-env, supports partial update with env_ids.

    Policy:
      - from: sample among occupied slots
      - to:   sample among empty slots
      - fallback if degenerate:
          * no occupied -> uniform over all slots
          * no empty    -> uniform over all slots != from
    """
    assert num_slots == 4, "This implementation assumes 4 slots"
    ensure_command_buffer(env)

    # Decide which envs to update
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
        if env_ids.numel() == 0:
            return env.command_from_to

    # Nearest-slot occupancy for ALL envs, then select env_ids
    nearest_all = _nearest_slot_for_each_cube_xy(env)          # (N,3)
    nearest = nearest_all.index_select(0, env_ids)             # (M,3)
    M = int(env_ids.numel())

    # occupancy multi-hot (M,4)
    occ = torch.zeros((M, num_slots), dtype=torch.bool, device=env.device)
    occ.scatter_(1, nearest, True)
    empty = ~occ

    # ---- sample FROM among occupied ----
    probs_from = occ.to(torch.float32)                         # (M,4)
    sum_from = probs_from.sum(dim=1, keepdim=True)             # (M,1)

    deg_from = (sum_from.squeeze(1) == 0)
    probs_from = torch.where(deg_from.unsqueeze(1), torch.ones_like(probs_from), probs_from)

    from_idx = torch.multinomial(probs_from, num_samples=1, replacement=True).squeeze(1)  # (M,)

    # ---- sample TO among empty ----
    probs_to = empty.to(torch.float32)                         # (M,4)
    sum_to = probs_to.sum(dim=1, keepdim=True)                 # (M,1)
    deg_to = (sum_to.squeeze(1) == 0)

    fallback_to = torch.ones_like(probs_to)
    fallback_to.scatter_(1, from_idx.view(-1, 1), 0.0)         # all slots except from

    probs_to = torch.where(deg_to.unsqueeze(1), fallback_to, probs_to)
    to_idx = torch.multinomial(probs_to, num_samples=1, replacement=True).squeeze(1)     # (M,)

    # Write ONLY in env_ids
    env.command_from_to[env_ids, 0] = from_idx + 1  # 1..4
    env.command_from_to[env_ids, 1] = to_idx + 1

    return env.command_from_to
