# mdp/commands.py
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from typing import TYPE_CHECKING

from .terminations import move_success

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _slots_w(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Slots in world frame.

    Returns:
        - (N,4,3) if env_ids is None
        - (M,4,3) if env_ids provided
    """
    slots_local = torch.as_tensor(env.cfg.slot_positions, dtype=torch.float32, device=env.device)  # (4,3)
    assert slots_local.shape == (4, 3), f"Expected slot_positions shape (4,3), got {slots_local.shape}"

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    origins = env.scene.env_origins.index_select(0, env_ids)  # (M,3)
    return slots_local.unsqueeze(0) + origins.unsqueeze(1)    # (M,4,3)



CUBE_KEYS_9 = [
    "cube_light_s", "cube_light_m", "cube_light_l",
    "cube_flat_s",  "cube_flat_m",  "cube_flat_l",
    "cube_dark_s",  "cube_dark_m",  "cube_dark_l",
]

def _active_cube_positions_w(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """
    Active cube positions in world frame.

    Returns:
        - (N,3,3) if env_ids is None
        - (M,3,3) if env_ids provided
    """
    assert hasattr(env, "active_cube_indices"), "env.active_cube_indices missing. Call reset event randomize_cubes_on_slots first."

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    pos9_all = torch.stack([env.scene[k].data.root_pos_w for k in CUBE_KEYS_9], dim=1)  # (N,9,3)
    pos9 = pos9_all.index_select(0, env_ids)  # (M,9,3)

    idx = env.active_cube_indices.index_select(0, env_ids)  # (M,3)
    return pos9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (M,3,3)

# Replace old uses:
# _cube_positions_w(...) -> _active_cube_positions_w(env)
def _nearest_slot_for_each_cube_xy(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Nearest slot index (0..3) for each active cube, per env."""
    cubes = _active_cube_positions_w(env, env_ids=env_ids)[:, :, :2]  # (M,3,2)
    slots = _slots_w(env, env_ids=env_ids)[:, :, :2]                  # (M,4,2)

    d = cubes.unsqueeze(2) - slots.unsqueeze(1)   # (M,3,4,2)
    dist2 = (d * d).sum(dim=-1)                   # (M,3,4)
    return torch.argmin(dist2, dim=2)             # (M,3)


def ensure_command_buffer(env: ManagerBasedRLEnv):
    """Create env.command_from_to if it doesn't exist."""
    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        env.command_from_to = torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)

def ensure_moc_buffers(env: ManagerBasedRLEnv):
    """Auxiliary buffers for Phase-1 correctness/stability."""
    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        env.target_cube_id = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    if not hasattr(env, "moc_cmd_cube_pos_xy0") or env.moc_cmd_cube_pos_xy0 is None:
        env.moc_cmd_cube_pos_xy0 = torch.zeros((env.num_envs, 3, 2), dtype=torch.float32, device=env.device)
    if not hasattr(env, "moc_cmd_stamp") or env.moc_cmd_stamp is None:
        env.moc_cmd_stamp = -torch.ones((env.num_envs,), dtype=torch.long, device=env.device)
    
    # --- Phase machine buffers (per-env) ---
    if not hasattr(env, "moc_phase") or env.moc_phase is None:
        # phases: 1=approach, 2=suction_hold, 3=lift_move_place, 4=release_hold, 5=ready_next
        env.moc_phase = torch.ones((env.num_envs,), dtype=torch.int32, device=env.device)

    if not hasattr(env, "moc_phase_hold") or env.moc_phase_hold is None:
        # small counter to require conditions for a few consecutive steps
        env.moc_phase_hold = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    if not hasattr(env, "moc_prev_ep_len") or env.moc_prev_ep_len is None:
        # to detect episode reset (ep_len decreases to 0)
        env.moc_prev_ep_len = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    # --- NEXT edge detect (prevents multi-fire while holding the button) ---
    if not hasattr(env, "moc_next_prev") or env.moc_next_prev is None:
        env.moc_next_prev = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)


def latch_command_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """Latch per-command auxiliary state without resampling the command.

    This sets (for env_ids):
      - env.target_cube_id          (N,) in {0,1,2} for active cubes
      - env.moc_cmd_cube_pos_xy0    (N,3,2) baseline XY of active cubes
      - env.moc_cmd_stamp           (N,) episode step stamp (if available)

    Use this when you set commands manually or when buffers were not initialized.
    """
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
        if env_ids.numel() == 0:
            return

    # Need current command + current cube/slot state
    cmd = env.command_from_to.index_select(0, env_ids).to(device=env.device)
    from_idx = torch.clamp(cmd[:, 0] - 1, 0, 3)  # (M,)

    cubes_pos = _active_cube_positions_w(env, env_ids=env_ids)  # (M,3,3)
    slots = _slots_w(env, env_ids=env_ids)                      # (M,4,3)

    nearest = _nearest_slot_for_each_cube_xy(env).index_select(0, env_ids)  # (M,3)
    match = (nearest == from_idx.unsqueeze(1))                              # (M,3)
    has_match = match.any(dim=1)                                           # (M,)
    first_match_id = match.to(torch.int64).argmax(dim=1)                   # (M,)

    from_slot_xy = slots[torch.arange(env_ids.numel(), device=env.device), from_idx, :2]  # (M,2)
    dxy = cubes_pos[:, :, :2] - from_slot_xy.unsqueeze(1)                                  # (M,3,2)
    dist2_from = (dxy * dxy).sum(dim=-1)                                                    # (M,3)
    fallback_id = dist2_from.argmin(dim=1)                                                  # (M,)

    target_id = torch.where(has_match, first_match_id, fallback_id)                         # (M,)

    env.target_cube_id[env_ids] = target_id
    env.moc_cmd_cube_pos_xy0[env_ids] = cubes_pos[:, :, :2]
    if hasattr(env, "episode_length_buf"):
        env.moc_cmd_stamp[env_ids] = env.episode_length_buf[env_ids]


def set_command_from_to(env: ManagerBasedRLEnv, from_slot_1based: int, to_slot_1based: int):
    """Set same command for all envs (debug) AND latch target/baselines."""
    ensure_command_buffer(env)
    ensure_moc_buffers(env)
    env.command_from_to[:, 0] = int(from_slot_1based)
    env.command_from_to[:, 1] = int(to_slot_1based)
    latch_command_state(env)


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
    ensure_moc_buffers(env)

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


    # ---- choose TO as an actually empty slot (deterministic when unique) ----
    empty_count = empty.sum(dim=1)  # (M,) expected == 1 with 3 cubes / 4 slots

    # Case A: exactly one empty -> pick it deterministically
    to_idx_unique = empty.to(torch.int64).argmax(dim=1)  # (M,)

    # Case B: multiple empties (degenerate occupancy) -> pick uniformly among empties
    # Build probs over empties; if none empty -> fallback to any slot != from
    # ---- choose TO: prefer the unique empty slot (O(1), no loops) ----
    empty_count = empty.sum(dim=1)  # (M,) expected == 1 with 3 cubes / 4 slots

    # Case A: exactly one empty -> pick it deterministically
    to_idx_unique = empty.to(torch.int64).argmax(dim=1)  # (M,)

    # Case B: multiple empties (degenerate occupancy) -> pick among empties but never equal to from
    probs_to = empty.to(torch.float32)  # (M,4)
    probs_to.scatter_(1, from_idx.view(-1, 1), 0.0)  # forbid to==from

    # Case C: if no valid empty remains (all mass removed), fallback to any slot != from
    fallback_to = torch.ones_like(probs_to)
    fallback_to.scatter_(1, from_idx.view(-1, 1), 0.0)

    valid_to_count = probs_to.sum(dim=1)  # (M,)
    probs_to = torch.where(valid_to_count.unsqueeze(1) > 0, probs_to, fallback_to)

    to_idx_multi = torch.multinomial(probs_to, num_samples=1, replacement=True).squeeze(1)
    
    # Deterministic when we have the unique empty slot, otherwise sampled
    to_idx = torch.where(empty_count == 1, to_idx_unique, to_idx_multi)


    # ---- latch target_cube_id for this command (per-env) ----
    cubes_pos = _active_cube_positions_w(env, env_ids=env_ids)          # (M,3,3)
    slots_w = _slots_w(env, env_ids=env_ids)                            # (M,4,3)

    match = (nearest == from_idx.unsqueeze(1))                          # (M,3)
    has_match = match.any(dim=1)                                        # (M,)
    first_match_id = match.to(torch.int64).argmax(dim=1)                # (M,)

    from_slot_xy = slots_w[torch.arange(M, device=env.device), from_idx, :2]  # (M,2)
    dxy = cubes_pos[:, :, :2] - from_slot_xy.unsqueeze(1)               # (M,3,2)
    dist2_from = (dxy * dxy).sum(dim=-1)                                # (M,3)
    fallback_id = dist2_from.argmin(dim=1)                              # (M,)

    target_id = torch.where(has_match, first_match_id, fallback_id)     # (M,)

    env.target_cube_id[env_ids] = target_id
    env.moc_cmd_cube_pos_xy0[env_ids] = cubes_pos[:, :, :2]
    if hasattr(env, "episode_length_buf"):
        env.moc_cmd_stamp[env_ids] = env.episode_length_buf[env_ids]


    # Write ONLY in env_ids
    env.command_from_to[env_ids, 0] = from_idx + 1  # 1..4
    env.command_from_to[env_ids, 1] = to_idx + 1

    return env.command_from_to


@torch.no_grad()
def resample_commands_until_not_success(
    env,
    max_tries: int = 20,
    tol_xy: float = 0.02,
    tol_z: float = 0.05,
    clear_tol_xy: float = 0.08,
) -> None:
    """Resample command_from_to on reset so that move_success is not already True.

    Uses require_settled=False to avoid dependence on velocity thresholds.
    """
    # Nota: asumimos que active_cube_indices ya existe (event randomize_cubes ejecutado).

    for _ in range(max_tries):
        # sample_command_from_to ya debe rellenar env.command_from_to y latchear target/baseline
        sample_command_from_to(env)  # <-- esta función YA existe en commands.py

        # si tu sample_command_from_to NO hace latch, descomenta:
        # latch_command_state(env)

        ms = move_success(
            env,
            tol_xy=tol_xy,
            tol_z=tol_z,
            require_to_clear=True,
            clear_tol_xy=clear_tol_xy,
            require_settled=False,
        )

        if not bool(ms.any().item()):
            return

    # Si llegamos aquí, seguimos teniendo algunos success; forzamos "from!=to" y salimos.
    # (Dejar claro en log para depuración.)
    print(
        f"[WARN] resample_commands_until_not_success reached max_tries={max_tries}; "
        f"some envs may start in success."
    )