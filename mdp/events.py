from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import latch_target_cube_from_command, sample_command_from_to
from .constants import CUBE_KEYS_9

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_COLOR_BASE = torch.tensor([0, 3, 6], dtype=torch.long)


def _maybe_set_visibility(cube, visible: bool, env_ids: torch.Tensor) -> None:
    if hasattr(cube, "set_visibility"):
        try:
            cube.set_visibility(visible, env_ids=env_ids)
        except Exception:
            pass


def randomize_cubes_on_slots(env: "ManagerBasedRLEnv", env_ids) -> None:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return

    num_envs = env_ids.numel()
    num_slots = 4
    num_active_cubes = 3

    slots_local = torch.as_tensor(env.cfg.slot_positions, dtype=torch.float32, device=env.device)
    origins = env.scene.env_origins.index_select(0, env_ids)
    slots_w = origins.unsqueeze(1) + slots_local.unsqueeze(0)

    active_slot_idx = torch.rand((num_envs, num_slots), device=env.device).argsort(dim=1)[:, :num_active_cubes]
    active_cube_pos_w = slots_w.gather(1, active_slot_idx.unsqueeze(-1).expand(-1, -1, 3))

    quat_identity = torch.zeros((num_envs, 4), dtype=torch.float32, device=env.device)
    quat_identity[:, 0] = 1.0
    zero_vel = torch.zeros((num_envs, 6), dtype=torch.float32, device=env.device)

    size_perm = torch.rand((num_envs, 3), device=env.device).argsort(dim=1)
    active_cube_indices = _COLOR_BASE.to(env.device).view(1, 3) + size_perm

    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        env.active_cube_indices = torch.zeros((env.num_envs, 3), dtype=torch.long, device=env.device)
    env.active_cube_indices[env_ids] = active_cube_indices

    if not hasattr(env, "moc_active_cube_slot_idx") or env.moc_active_cube_slot_idx is None:
        env.moc_active_cube_slot_idx = torch.zeros((env.num_envs, 3), dtype=torch.long, device=env.device)

    if not hasattr(env, "moc_slot_to_active_id") or env.moc_slot_to_active_id is None:
        env.moc_slot_to_active_id = -torch.ones((env.num_envs, 4), dtype=torch.long, device=env.device)

    env.moc_active_cube_slot_idx[env_ids] = active_slot_idx.to(torch.long)
    env.moc_slot_to_active_id[env_ids] = -1

    for active_id in range(3):
        slot_idx = active_slot_idx[:, active_id].to(torch.long)
        env.moc_slot_to_active_id[env_ids, slot_idx] = active_id

    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        env.target_cube_id = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    env.target_cube_id[env_ids] = 0

    for cube_idx, cube_key in enumerate(CUBE_KEYS_9):
        cube = env.scene[cube_key]

        is_active = (active_cube_indices == cube_idx).any(dim=1)
        if is_active.any():
            active_env_ids = env_ids[is_active]
            which_col = (active_cube_indices[is_active] == cube_idx).to(torch.int64).argmax(dim=1)

            pose = torch.cat(
                [
                    active_cube_pos_w[is_active]
                    .gather(1, which_col.view(-1, 1, 1).expand(-1, 1, 3))
                    .squeeze(1),
                    quat_identity[is_active],
                ],
                dim=1,
            )

            cube.write_root_pose_to_sim(pose, env_ids=active_env_ids)
            cube.write_root_velocity_to_sim(zero_vel[is_active], env_ids=active_env_ids)
            _maybe_set_visibility(cube, True, active_env_ids)

        is_inactive = ~is_active
        if is_inactive.any():
            inactive_env_ids = env_ids[is_inactive]
            y_off = (float(cube_idx) - 4.0) * 0.25
            parked_pos = origins[is_inactive] + torch.tensor([5.0, y_off, 0.20], dtype=torch.float32, device=env.device)
            parked_pose = torch.cat([parked_pos, quat_identity[is_inactive]], dim=1)

            cube.write_root_pose_to_sim(parked_pose, env_ids=inactive_env_ids)
            cube.write_root_velocity_to_sim(zero_vel[is_inactive], env_ids=inactive_env_ids)
            _maybe_set_visibility(cube, False, inactive_env_ids)


def moc_reset_on_reset(env: "ManagerBasedRLEnv", env_ids=None) -> None:
    if not hasattr(env, "_moc_reset_id") or env._moc_reset_id is None:
        env._moc_reset_id = 0
    env._moc_reset_id += 1

    randomize_cubes_on_slots(env, env_ids)

    try:
        if hasattr(env.scene, "write_data_to_sim"):
            env.scene.write_data_to_sim()
        if hasattr(env, "sim") and hasattr(env.sim, "step"):
            env.sim.step()
        if hasattr(env.scene, "update"):
            try:
                env.scene.update()
            except TypeError:
                pass
    except Exception:
        pass

    sample_command_from_to(env, env_ids=env_ids)
    latch_target_cube_from_command(env, env_ids)