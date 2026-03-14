from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .step_cache import get_active_cube_pos_w, get_slots_w, get_tcp_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_ids(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device)


def _target_cube_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    cubes = get_active_cube_pos_w(env)
    row = _env_ids(env)

    if hasattr(env, "target_cube_id") and env.target_cube_id is not None:
        target_id = env.target_cube_id.to(torch.long).clamp(0, cubes.shape[1] - 1)
    else:
        target_id = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

    return cubes[row, target_id, :]


def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1) + eps)


def reward_reach_xy_rational(
    env: "ManagerBasedRLEnv",
    k_xy: float = 0.10,
    p: float = 1.0,
) -> torch.Tensor:
    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    cube = _target_cube_pos_w(env)

    dist_xy = _safe_norm(tip[:, :2] - cube[:, :2])

    k = float(max(1e-6, k_xy))
    p = float(max(1e-3, p))
    reward = 1.0 / (1.0 + torch.pow(dist_xy / k, p))

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/reach_dist_xy"] = dist_xy

    return reward


def reward_reach_xy_progress(
    env: "ManagerBasedRLEnv",
    scale: float = 1.0,
    clip: float = 0.02,
) -> torch.Tensor:
    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    cube = _target_cube_pos_w(env)

    dist_xy = _safe_norm(tip[:, :2] - cube[:, :2])

    if not hasattr(env, "_moc_prev_reach_dist_xy") or env._moc_prev_reach_dist_xy is None:
        env._moc_prev_reach_dist_xy = dist_xy.detach()

    prev_dist = env._moc_prev_reach_dist_xy
    if hasattr(env, "reset_buf") and env.reset_buf is not None:
        reset_mask = env.reset_buf.to(torch.bool)
        prev_dist = torch.where(reset_mask, dist_xy.detach(), prev_dist)

    delta = torch.clamp(prev_dist - dist_xy, -float(max(1e-6, clip)), float(max(1e-6, clip)))
    env._moc_prev_reach_dist_xy = dist_xy.detach()

    reward = float(scale) * delta

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/reach_delta_xy"] = reward

    return reward


def reward_reach_z_gated(
    env: "ManagerBasedRLEnv",
    sigma_z: float = 0.06,
    gate_dxy: float = 0.18,
    gate_band: float = 0.05,
) -> torch.Tensor:
    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    cube = _target_cube_pos_w(env)

    dist_xy = _safe_norm(tip[:, :2] - cube[:, :2])

    dz = tip[:, 2] - cube[:, 2]
    sigma = float(max(1e-6, sigma_z))
    z_reward = torch.exp(-0.5 * (dz * dz) / (sigma * sigma))

    gate = torch.sigmoid((float(gate_dxy) - dist_xy) / float(max(1e-6, gate_band)))

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/reach_gate_xy"] = gate
    env.extras["moc/reach_abs_dz"] = torch.abs(dz)

    return gate * z_reward


def penalty_arm_joint_velocity(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    joint_names: list[str] | None = None,
) -> torch.Tensor:

    robot = env.scene[asset_name]

    joint_ids, _ = robot.find_joints(joint_names)

    qd = robot.data.joint_vel[:, joint_ids]

    penalty = torch.sum(qd * qd, dim=-1)

    env.extras["moc/arm_joint_vel_l2"] = penalty

    return penalty