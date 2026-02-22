# mdp/terminations.py
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import FrameTransformer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _slots_w(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    slots_local = torch.as_tensor(env.cfg.slot_positions, dtype=torch.float32, device=env.device)  # (4,3)
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
    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        raise RuntimeError("env.active_cube_indices missing. Ensure reset event randomize_cubes_on_slots runs before termination.")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
    pos9_all = torch.stack([env.scene[k].data.root_pos_w for k in CUBE_KEYS_9], dim=1)  # (N,9,3)
    pos9 = pos9_all.index_select(0, env_ids)  # (M,9,3)
    idx = env.active_cube_indices.index_select(0, env_ids)  # (M,3)
    return pos9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (M,3,3)

def _nearest_slot_for_each_cube_xy(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    cubes = _active_cube_positions_w(env, env_ids=env_ids)[:, :, :2]  # (M,3,2)
    slots = _slots_w(env, env_ids=env_ids)[:, :, :2]                  # (M,4,2)
    d = cubes.unsqueeze(2) - slots.unsqueeze(1)
    dist2 = (d * d).sum(dim=-1)
    return torch.argmin(dist2, dim=2)  # (M,3)

def _is_gripper_open(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # -------------------------
    # suction gripper (surface gripper)
    # -------------------------
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        sg = env.scene.surface_grippers["surface_gripper"]
        # En Isaac Lab, típicamente state == -1 indica "no agarrando" (abierto)
        return (sg.state.view(-1) == -1)

    # -------------------------
    # parallel gripper (2 joints)
    # -------------------------
    robot: Articulation = env.scene[robot_cfg.name]

    # Si no hay nombres configurados, no bloquees terminaciones por gripper
    if not hasattr(env.cfg, "gripper_joint_names"):
        return torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)

    # Cache de joint_ids para no llamar find_joints en cada step
    cache_attr = "_moc_gripper_joint_ids"
    joint_ids = getattr(env, cache_attr, None)
    if joint_ids is None:
        joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        setattr(env, cache_attr, joint_ids)

    # Si no son 2 joints, no bloquees por gripper (fallback safe)
    if len(joint_ids) != 2:
        return torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)

    open_val = torch.as_tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)

    # joint_pos: (N, num_joints)
    jp = robot.data.joint_pos
    ok1 = torch.isclose(jp[:, joint_ids[0]], open_val, atol=1e-3, rtol=1e-3)
    ok2 = torch.isclose(jp[:, joint_ids[1]], open_val, atol=1e-3, rtol=1e-3)
    return ok1 & ok2


def move_success(env: ManagerBasedRLEnv, tol_xy: float = 0.02, tol_z: float = 0.05) -> torch.Tensor:
    """Success if the *target cube* (latched if available) is at to_slot.

    Returns:
        (N,) bool tensor
    """
    # Command is stored 1..4
    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    cmd = env.command_from_to.to(device=env.device)
    from_idx = torch.clamp(cmd[:, 0] - 1, 0, 3)  # (N,)
    to_idx = torch.clamp(cmd[:, 1] - 1, 0, 3)    # (N,)

    cubes_pos = _active_cube_positions_w(env)  # (N,3,3)
    slots = _slots_w(env)                      # (N,4,3)

    # --- Determine target_cube_id ---
    if hasattr(env, "target_cube_id") and env.target_cube_id is not None:
        target_cube_id = torch.clamp(env.target_cube_id.to(device=env.device), 0, 2)  # (N,)
    else:
        # Infer target cube as the one currently assigned to from_slot (nearest in XY).
        nearest = _nearest_slot_for_each_cube_xy(env)  # (N,3)

        match = (nearest == from_idx.unsqueeze(1))     # (N,3)
        has_match = match.any(dim=1)                   # (N,)
        first_match_id = match.to(torch.int64).argmax(dim=1)

        from_slot_xy = slots[torch.arange(env.num_envs, device=env.device), from_idx, :2]  # (N,2)
        dxy = cubes_pos[:, :, :2] - from_slot_xy.unsqueeze(1)                                # (N,3,2)
        dist2 = (dxy * dxy).sum(dim=-1)                                                      # (N,3)
        fallback_id = dist2.argmin(dim=1)                                                    # (N,)

        target_cube_id = torch.where(has_match, first_match_id, fallback_id)                 # (N,)

    # --- Check target cube near to_slot ---
    target_pos = cubes_pos[torch.arange(env.num_envs, device=env.device), target_cube_id, :]  # (N,3)
    target_slot_pos = slots[torch.arange(env.num_envs, device=env.device), to_idx, :]         # (N,3)

    dx = target_pos[:, 0] - target_slot_pos[:, 0]
    dy = target_pos[:, 1] - target_slot_pos[:, 1]
    dz = target_pos[:, 2] - target_slot_pos[:, 2]

    ok_xy = (dx * dx + dy * dy) <= (float(tol_xy) ** 2)
    ok_z = torch.abs(dz) <= float(tol_z)

    return ok_xy & ok_z