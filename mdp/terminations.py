# mdp/terminations.py
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import FrameTransformer
from typing import TYPE_CHECKING

import re

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .step_cache import (
    get_slots_w,
    get_active_cube_pos_w,
    get_nearest_slot_for_active_cubes_xy,
    CUBE_KEYS_9
)


def time_out(env) -> torch.Tensor:
    """Time-limit truncation. Returns per-env bool tensor."""

    if not hasattr(env, "episode_length_buf"):
        raise AttributeError("env.episode_length_buf not found; required for time_out termination.")
    if not hasattr(env, "max_episode_length"):
        raise AttributeError("env.max_episode_length not found; required for time_out termination.")

    # Truncate on last step
    return env.episode_length_buf >= (int(env.max_episode_length) - 1)

def cube_fell_off_table(
    env: "ManagerBasedRLEnv",
    z_margin_below_slots: float = 0.15,
    xy_margin: float = 0.35,
) -> torch.Tensor:
    """
    Falla si cualquier cubo activo se ha caído de la mesa o está claramente fuera de la zona útil.

    - Usa slots como referencia (robusto a offsets).
    - Criterios:
      * z < (min_slot_z - z_margin_below_slots)
      * x o y fuera del bounding box de slots ± xy_margin
    """
    cubes_pos = get_active_cube_pos_w(env)  # (N,3,3)
    slots = get_slots_w(env)               # (N,4,3)

    # Umbral Z: debajo de la superficie de slots (aprox mesa)
    min_slot_z = slots[:, :, 2].min(dim=1).values  # (N,)
    z_thresh = min_slot_z - float(z_margin_below_slots)

    z_bad = (cubes_pos[:, :, 2] < z_thresh.unsqueeze(1)).any(dim=1)  # (N,)

    # Bounding box XY de slots
    min_xy = slots[:, :, :2].min(dim=1).values  # (N,2)
    max_xy = slots[:, :, :2].max(dim=1).values  # (N,2)

    min_xy = min_xy - float(xy_margin)
    max_xy = max_xy + float(xy_margin)

    x = cubes_pos[:, :, 0]
    y = cubes_pos[:, :, 1]

    x_bad = ((x < min_xy[:, 0].unsqueeze(1)) | (x > max_xy[:, 0].unsqueeze(1))).any(dim=1)
    y_bad = ((y < min_xy[:, 1].unsqueeze(1)) | (y > max_xy[:, 1].unsqueeze(1))).any(dim=1)

    xy_bad = x_bad | y_bad

    return z_bad | xy_bad


def ee_below_table(
    env: "ManagerBasedRLEnv",
    table_z: float = 0.0199,
    z_margin_below_slots: float = 0.002,
) -> torch.Tensor:
    """
    Terminates if Tip (ee_frame) goes below table threshold.
    We REQUIRE ee_frame to exist to avoid silent dead terminations.
    """
    try:
        ee_frame = env.scene["ee_frame"]
    except KeyError as e:
        raise RuntimeError(
            "ee_below_table: env.scene['ee_frame'] missing. "
            "Check moc_ur10_env_cfg.py: ee_frame must target '{ENV_REGEX_NS}/Robot/ee_link/Tip'."
        ) from e

    tip_pos = ee_frame.data.target_pos_w[:, 0, :3]
    z_thresh = float(table_z) - float(z_margin_below_slots)
    return tip_pos[:, 2] < z_thresh


def _active_cube_lin_vel_w(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        raise RuntimeError("env.active_cube_indices missing. Ensure reset event randomize_cubes_on_slots runs before termination.")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
    vel9_all = torch.stack([env.scene[k].data.root_lin_vel_w for k in CUBE_KEYS_9], dim=1)  # (N,9,3)
    vel9 = vel9_all.index_select(0, env_ids)  # (M,9,3)
    idx = env.active_cube_indices.index_select(0, env_ids)  # (M,3)
    return vel9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (M,3,3)


def _active_cube_ang_vel_w(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        raise RuntimeError("env.active_cube_indices missing. Ensure reset event randomize_cubes_on_slots runs before termination.")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
    vel9_all = torch.stack([env.scene[k].data.root_ang_vel_w for k in CUBE_KEYS_9], dim=1)  # (N,9,3)
    vel9 = vel9_all.index_select(0, env_ids)  # (M,9,3)
    idx = env.active_cube_indices.index_select(0, env_ids)  # (M,3)
    return vel9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (M,3,3)



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
    tol_xy: float = 0.02,
    tol_z: float = 0.05,
    require_to_clear: bool = True,
    clear_tol_xy: float | None = None,
    require_settled: bool = False,
    vel_tol: float = 0.20,
) -> torch.Tensor:
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

    cubes_pos = get_active_cube_pos_w(env)  # (N,3,3)
    slots = get_slots_w(env)               # (N,4,3)

    # --- Determine target_cube_id ---
    if hasattr(env, "target_cube_id") and env.target_cube_id is not None:
        target_cube_id = torch.clamp(env.target_cube_id.to(device=env.device), 0, 2)  # (N,)
    else:
        # Infer target cube as the one currently assigned to from_slot (nearest in XY).
        nearest = get_nearest_slot_for_active_cubes_xy(env)  # (N,3)

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

    ok = ok_xy & ok_z

    # --- Optional: require that no *other* cube occupies the to_slot region ---
    if require_to_clear:
        if clear_tol_xy is None:
            clear_tol_xy = max(float(tol_xy) * 1.5, 0.03)

        dxy_all = cubes_pos[:, :, :2] - target_slot_pos[:, :2].unsqueeze(1)  # (N,3,2)
        dist2_all = (dxy_all * dxy_all).sum(dim=-1)                          # (N,3)

        mask_other = torch.ones_like(dist2_all, dtype=torch.bool)
        mask_other.scatter_(1, target_cube_id.view(-1, 1), False)

        other_close = (dist2_all <= float(clear_tol_xy) ** 2) & mask_other
        ok = ok & (~other_close.any(dim=1))

    # --- Optional: require that target cube is settled (low linear+angular velocity) ---
    if require_settled:
        lin = _active_cube_lin_vel_w(env)  # (N,3,3)
        ang = _active_cube_ang_vel_w(env)  # (N,3,3)
        lin_t = lin[torch.arange(env.num_envs, device=env.device), target_cube_id, :]
        ang_t = ang[torch.arange(env.num_envs, device=env.device), target_cube_id, :]
        lin_n = torch.linalg.vector_norm(lin_t, dim=-1)
        ang_n = torch.linalg.vector_norm(ang_t, dim=-1)
        ok = ok & (lin_n <= float(vel_tol)) & (ang_n <= float(vel_tol))

    return ok