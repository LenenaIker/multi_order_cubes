# mdp/terminations.py
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import FrameTransformer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _slots_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    slots = torch.as_tensor(env.cfg.slot_positions, dtype=torch.float32, device=env.device)
    return slots

CUBE_KEYS_9 = [
    "cube_light_s", "cube_light_m", "cube_light_l",
    "cube_flat_s",  "cube_flat_m",  "cube_flat_l",
    "cube_dark_s",  "cube_dark_m",  "cube_dark_l",
]

def _active_cube_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        # Default: pick the "medium" instance for each color => indices [1, 4, 7]
        env.active_cube_indices = torch.zeros((env.num_envs, 3), dtype=torch.long, device=env.device)
        env.active_cube_indices[:, 0] = 1  # light_m
        env.active_cube_indices[:, 1] = 4  # flat_m
        env.active_cube_indices[:, 2] = 7  # dark_m
    pos9 = torch.stack([env.scene[k].data.root_pos_w for k in CUBE_KEYS_9], dim=1)  # (N,9,3)
    idx = env.active_cube_indices  # (N,3)
    return pos9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (N,3,3)

def _nearest_slot_for_each_cube_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    cubes = _active_cube_positions_w(env)[:, :, :2]  # (N,3,2)
    slots = _slots_w(env)[:, :2].unsqueeze(0)        # (1,4,2)
    d = cubes.unsqueeze(2) - slots.unsqueeze(1)
    dist2 = (d * d).sum(dim=-1)
    return torch.argmin(dist2, dim=2)  # (N,3)

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


def move_success(
    env: ManagerBasedRLEnv,
    xy_tol: float = 0.03,
    require_gripper_open: bool = True,
) -> torch.Tensor:
    """
    Success if the cube that STARTED at 'from' is now at 'to' (nearest-slot wise + XY tolerance),
    and (optionally) the gripper is open.
    """
    assert hasattr(env, "command_from_to"), "env.command_from_to not found. Add commands manager or set it manually."
    cmd = env.command_from_to  # (N,2) values 1..4
    from_idx = torch.clamp(cmd[:, 0] - 1, 0, 3)
    to_idx = torch.clamp(cmd[:, 1] - 1, 0, 3)

    nearest = _nearest_slot_for_each_cube_xy(env)  # (N,3)
    cubes_pos = _active_cube_positions_w(env)  # (N,3,3)
    slots = _slots_w(env)  # (4,3)

    # Identify target cube as the cube whose nearest slot == from_idx.
    # If none matches (degenerate), pick the cube closest to the 'from' slot in XY.
    match = (nearest == from_idx.unsqueeze(1))  # (N,3)
    has_match = match.any(dim=1)  # (N,)
    first_match_id = match.to(torch.int64).argmax(dim=1)  # (N,)

    from_slot_xy = slots[from_idx, :2]  # (N,2)
    d = cubes_pos[:, :, :2] - from_slot_xy.unsqueeze(1)  # (N,3,2)
    dist2 = (d * d).sum(dim=-1)  # (N,3)
    fallback_id = dist2.argmin(dim=1)  # (N,)

    target_cube_id = torch.where(has_match, first_match_id, fallback_id)

    # distance of target cube to 'to' slot
    target_pos = cubes_pos[torch.arange(env.num_envs, device=env.device), target_cube_id, :]
    target_slot_pos = slots[to_idx]  # (N,3) via fancy indexing
    xy_dist = torch.linalg.vector_norm(target_pos[:, :2] - target_slot_pos[:, :2], dim=1)

    ok = xy_dist < xy_tol
    if require_gripper_open:
        ok = ok & _is_gripper_open(env)
    return ok
