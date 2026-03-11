from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .step_cache import (
    get_active_cube_pos_w,
    get_active_cube_quat_w,
    get_nearest_slot_for_active_cubes_xy,
    get_slots_w,
    get_tcp_pose_w,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def slot_positions_in_base_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    slots_w = get_slots_w(env)

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=env.device)
    slots_quat_w = ident.view(1, 1, 4).expand(env.num_envs, 4, 4)

    slots_pos_b, _ = math_utils.subtract_frame_transforms(
        root_pos_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3),
        root_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),
        slots_w.reshape(-1, 3),
        slots_quat_w.reshape(-1, 4),
    )
    return slots_pos_b.view(env.num_envs, 4, 3).reshape(env.num_envs, 12)


def command_from_to_onehot(env: "ManagerBasedRLEnv", num_slots: int = 4) -> torch.Tensor:
    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        return torch.zeros((env.num_envs, 2 * num_slots), dtype=torch.float32, device=env.device)

    cmd = env.command_from_to
    from_idx = torch.clamp(cmd[:, 0] - 1, 0, num_slots - 1)
    to_idx = torch.clamp(cmd[:, 1] - 1, 0, num_slots - 1)

    from_oh = torch.zeros((env.num_envs, num_slots), dtype=torch.float32, device=env.device)
    to_oh = torch.zeros((env.num_envs, num_slots), dtype=torch.float32, device=env.device)

    from_oh.scatter_(1, from_idx.view(-1, 1), 1.0)
    to_oh.scatter_(1, to_idx.view(-1, 1), 1.0)

    return torch.cat([from_oh, to_oh], dim=-1)


def cubes_slot_occupancy_onehot(env: "ManagerBasedRLEnv", num_slots: int = 4) -> torch.Tensor:
    if hasattr(env, "moc_slot_to_active_id") and env.moc_slot_to_active_id is not None:
        return (env.moc_slot_to_active_id >= 0).to(torch.float32)

    nearest = get_nearest_slot_for_active_cubes_xy(env, num_slots=num_slots)
    occ = torch.zeros((env.num_envs, num_slots), dtype=torch.float32, device=env.device)
    occ.scatter_(1, nearest, 1.0)
    return occ


def target_cube_pos_in_base_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)

    cubes_pos_w = get_active_cube_pos_w(env)
    target_id = env.target_cube_id.to(torch.long).clamp(0, 2)
    row = torch.arange(env.num_envs, device=env.device)
    target_pos_w = cubes_pos_w[row, target_id, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=env.device).view(1, 4).expand(env.num_envs, 4)
    target_pos_b, _ = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, ident)
    return target_pos_b


def ee_pose_in_base_frame(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    ee_pos_w, ee_quat_w = get_tcp_pose_w(env, ee_frame_name=ee_frame_cfg.name, quat_mode="avg")

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    if return_key == "pos":
        return ee_pos_b
    if return_key == "quat":
        return ee_quat_b
    return torch.cat([ee_pos_b, ee_quat_b], dim=1)


def cubes_poses_in_base_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    cubes_pos_w = get_active_cube_pos_w(env)
    cubes_quat_w = get_active_cube_quat_w(env)

    n = env.num_envs
    root_pos_rep = root_pos_w.unsqueeze(1).expand(n, 3, 3).reshape(-1, 3)
    root_quat_rep = root_quat_w.unsqueeze(1).expand(n, 3, 4).reshape(-1, 4)

    pos_b, quat_b = math_utils.subtract_frame_transforms(
        root_pos_rep,
        root_quat_rep,
        cubes_pos_w.reshape(-1, 3),
        cubes_quat_w.reshape(-1, 4),
    )

    return torch.cat([pos_b, quat_b], dim=1).view(n, 3, 7).reshape(n, 21)


def gripper_state(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(getattr(env.cfg, "gripper_joint_names", ["finger_joint"]))
    return robot.data.joint_pos[:, joint_ids[0]].to(torch.float32).unsqueeze(1)


def stable_success_hint(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "moc_stable_success") and env.moc_stable_success is not None:
        return env.moc_stable_success.to(torch.float32).view(-1, 1)
    return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)


def next_cooldown_obs(env: "ManagerBasedRLEnv", max_cooldown_steps: int = 30) -> torch.Tensor:
    if not hasattr(env, "moc_next_cooldown") or env.moc_next_cooldown is None:
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

    cd = env.moc_next_cooldown.to(torch.float32).clamp(min=0.0)
    return (cd / float(max(1, max_cooldown_steps))).unsqueeze(1)


def moc_phase_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "moc_phase") and env.moc_phase is not None:
        phase = env.moc_phase.to(torch.float32).clamp(1, 5)
        return ((phase - 1.0) / 4.0).view(-1, 1)
    return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)


def policy_obs(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return torch.cat(
        [
            cubes_poses_in_base_frame(env, robot_cfg=robot_cfg),
            target_cube_pos_in_base_frame(env, robot_cfg=robot_cfg),
            ee_pose_in_base_frame(env, ee_frame_cfg, robot_cfg),
            slot_positions_in_base_frame(env, robot_cfg),
            cubes_slot_occupancy_onehot(env, num_slots=4),
            command_from_to_onehot(env, num_slots=4),
            gripper_state(env, robot_cfg),
            stable_success_hint(env),
            moc_phase_obs(env),
            next_cooldown_obs(env),
        ],
        dim=1,
    )