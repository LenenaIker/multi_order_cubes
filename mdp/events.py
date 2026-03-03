# multi_order_cubes/mdp/events.py
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import torch

from .constants import CUBE_KEYS_9
from .commands import sample_command_from_to, latch_command_state



# indices helper:
# light: 0..2, flat: 3..5, dark: 6..8
_COLOR_BASE = torch.tensor([0, 3, 6], dtype=torch.long)


def sample_from_to_on_reset(env: "ManagerBasedRLEnv", env_ids=None):
    # Increment reset id to invalidate step_cache across resets even if no env.step() happens.
    if not hasattr(env, "_moc_reset_id") or env._moc_reset_id is None:
        env._moc_reset_id = 0
    env._moc_reset_id += 1

    # One-shot sampling (no retries). Command validity is by construction.
    sample_command_from_to(env, env_ids=env_ids)
    latch_command_state(env, env_ids)


def _maybe_set_visibility(cube, visible: bool, env_ids: torch.Tensor):
    """
    Best-effort visibility toggle. If API not available, do nothing.
    This does NOT touch materials/shading.
    """
    # Try common Isaac Lab wrappers
    if hasattr(cube, "set_visibility"):
        try:
            cube.set_visibility(visible, env_ids=env_ids)
            return
        except Exception:
            pass
    # If omni utils exist in your runtime, you can implement prim-level visibility here.
    # Keeping it as no-op to remain safe/portable.


def randomize_cubes_on_slots(env, env_ids):
    device = env.device
    env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    M = env_ids.numel()
    num_slots = 4
    num_cubes_active = 3

    # -------------------------
    # slots (world)
    # -------------------------
    slots_local = torch.as_tensor(env.cfg.slot_positions, device=device, dtype=torch.float32)  # (4,3)
    origins = env.scene.env_origins.index_select(0, env_ids)                                   # (M,3)
    slots_w = origins.unsqueeze(1) + slots_local.unsqueeze(0)                                  # (M,4,3)

    # 3 slots sin repetición por env
    perm_slots = torch.rand((M, num_slots), device=device).argsort(dim=1)[:, :num_cubes_active]  # (M,3)
    pos_active = slots_w.gather(1, perm_slots.unsqueeze(-1).expand(-1, -1, 3))                   # (M,3,3)

    quat = torch.zeros((M, 4), device=device, dtype=torch.float32)
    quat[:, 0] = 1.0
    zero_vel = torch.zeros((M, 6), device=device, dtype=torch.float32)

    # -------------------------
    # tamaño sin repetición por env: [0,1,2] permutado
    # 0->s, 1->m, 2->l
    # -------------------------
    perm_size = torch.rand((M, 3), device=device).argsort(dim=1)  # (M,3)

    # active cube indices into CUBE_KEYS_9 for each env, in color order [light, flat, dark]
    color_base = _COLOR_BASE.to(device=device).view(1, 3).expand(M, 3)  # (M,3)
    active_idx = color_base + perm_size                                  # (M,3), values in [0..8]

    # Store for other modules (commands/obs/terminations)
    # Shape: (num_envs, 3), values 0..8 into CUBE_KEYS_9
    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        env.active_cube_indices = torch.zeros((env.num_envs, 3), dtype=torch.long, device=device)
    env.active_cube_indices[env_ids] = active_idx

    # --- Ensure target cube id exists and is valid (fallback) ---
    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        env.target_cube_id = torch.zeros((env.num_envs,), dtype=torch.long, device=device)

    # Default to first active cube (index 0 within active set) for these envs
    env.target_cube_id[env_ids] = 0

    # -------------------------
    # apply: activate selected 3 (place on slots), park other 6
    # -------------------------
    # parked_pos = origins + torch.tensor([0.0, 0.0, -10.0], device=device, dtype=torch.float32).view(1, 3)  # (M,3)
    # parked_pose = torch.cat([parked_pos, quat], dim=1)  # (M,7)

    # Park inactive cubes away from workspace (avoid placing below groundplane half-space).
    # NOTE: we park in +X direction, spread in Y per cube_j to avoid inter-cube contact.

    # Build per-cube env subsets and write pose/vel
    # We loop 9 cubes; each cube is active in a subset of env_ids
    arange_M = torch.arange(M, device=device)

    for cube_j, key in enumerate(CUBE_KEYS_9):
        cube = env.scene[key]

        # which of the M envs want this cube active?
        is_active_j = (active_idx == cube_j).any(dim=1)  # (M,)
        if is_active_j.any():
            sub_env_ids = env_ids[is_active_j]

            # position index among the 3 actives (0..2) for each env in sub_env_ids
            # (light/flat/dark order matches columns 0..2)
            which_col = (active_idx[is_active_j] == cube_j).to(torch.int64).argmax(dim=1)  # (K,)

            pose = torch.cat([pos_active[is_active_j, :, :].gather(
                1, which_col.view(-1, 1, 1).expand(-1, 1, 3)
            ).squeeze(1), quat[is_active_j]], dim=1)  # (K,7)

            cube.write_root_pose_to_sim(pose, env_ids=sub_env_ids)
            cube.write_root_velocity_to_sim(zero_vel[is_active_j], env_ids=sub_env_ids)
            _maybe_set_visibility(cube, True, sub_env_ids)

        # inactive subset
        is_inactive_j = ~is_active_j
        if is_inactive_j.any():
            sub_env_ids = env_ids[is_inactive_j]

            # Park inactive cubes away from workspace (avoid placing below groundplane half-space).
            # Separate per cube_j in Y to avoid collisions between parked cubes.
            y_off = (float(cube_j) - 4.0) * 0.25  # [-1.0, +1.0] approx
            parked_pos = origins[is_inactive_j] + torch.tensor(
                [5.0, y_off, 0.20], device=device, dtype=torch.float32
            ).view(1, 3)

            parked_pose = torch.cat([parked_pos, quat[is_inactive_j]], dim=1)  # (K,7)

            cube.write_root_pose_to_sim(parked_pose, env_ids=sub_env_ids)
            cube.write_root_velocity_to_sim(zero_vel[is_inactive_j], env_ids=sub_env_ids)
            _maybe_set_visibility(cube, False, sub_env_ids)