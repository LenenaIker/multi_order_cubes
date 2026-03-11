from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .constants import CUBE_KEYS_9

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_cache(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "_moc_cache") or env._moc_cache is None:
        env._moc_cache = {}
    if not hasattr(env, "_moc_cache_token") or env._moc_cache_token is None:
        env._moc_cache_token = -1
    if not hasattr(env, "_moc_reset_id") or env._moc_reset_id is None:
        env._moc_reset_id = 0


def _step_token(env: "ManagerBasedRLEnv") -> int:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        return int(getattr(env, "_moc_reset_id", 0))

    el = env.episode_length_buf
    s = int(el.sum().item())
    mx = int(el.max().item())
    mn = int(el.min().item())

    rb = 0
    if hasattr(env, "reset_buf") and env.reset_buf is not None:
        rb = int(env.reset_buf.sum().item())

    rid = int(getattr(env, "_moc_reset_id", 0))
    base = (s << 32) ^ (mx << 16) ^ (mn << 8) ^ rb
    return (rid << 48) ^ base


def _invalidate_cache_if_needed(env: "ManagerBasedRLEnv") -> None:
    _ensure_cache(env)
    token = _step_token(env)
    if env._moc_cache_token != token:
        env._moc_cache.clear()
        env._moc_cache_token = token


def get_slots_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _invalidate_cache_if_needed(env)
    key = "slots_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    slots_local = torch.as_tensor(env.cfg.slot_positions, dtype=torch.float32, device=env.device)
    slots_w = slots_local.unsqueeze(0) + env.scene.env_origins.unsqueeze(1)

    env._moc_cache[key] = slots_w
    return slots_w


def get_cube_pos9_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _invalidate_cache_if_needed(env)
    key = "cube_pos9_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    pos9 = torch.stack([env.scene[key].data.root_pos_w for key in CUBE_KEYS_9], dim=1)
    env._moc_cache[key] = pos9
    return pos9


def get_cube_quat9_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _invalidate_cache_if_needed(env)
    key = "cube_quat9_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    quat9 = torch.stack([env.scene[key].data.root_quat_w for key in CUBE_KEYS_9], dim=1)
    env._moc_cache[key] = quat9
    return quat9


def get_active_cube_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        if int(getattr(env, "_moc_reset_id", 0)) == 0:
            return torch.zeros((env.num_envs, 3, 3), dtype=torch.float32, device=env.device)
        raise RuntimeError("env.active_cube_indices is missing after reset.")

    _invalidate_cache_if_needed(env)
    key = "active_cube_pos_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    pos9 = get_cube_pos9_w(env)
    idx = env.active_cube_indices
    active = pos9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

    env._moc_cache[key] = active
    return active


def get_active_cube_quat_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        if int(getattr(env, "_moc_reset_id", 0)) == 0:
            quat = torch.zeros((env.num_envs, 3, 4), dtype=torch.float32, device=env.device)
            quat[..., 0] = 1.0
            return quat
        raise RuntimeError("env.active_cube_indices is missing after reset.")

    _invalidate_cache_if_needed(env)
    key = "active_cube_quat_w"
    if key in env._moc_cache:
        return env._moc_cache[key]

    quat9 = get_cube_quat9_w(env)
    idx = env.active_cube_indices
    active = quat9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))

    env._moc_cache[key] = active
    return active


def get_nearest_slot_for_active_cubes_xy(env: "ManagerBasedRLEnv", num_slots: int = 4) -> torch.Tensor:
    _invalidate_cache_if_needed(env)
    key = f"nearest_slot_xy:{num_slots}"
    if key in env._moc_cache:
        return env._moc_cache[key]

    cubes_xy = get_active_cube_pos_w(env)[:, :, :2]
    slots_xy = get_slots_w(env)[:, :num_slots, :2]

    diff = cubes_xy.unsqueeze(2) - slots_xy.unsqueeze(1)
    dist2 = (diff * diff).sum(dim=-1)
    nearest = torch.argmin(dist2, dim=2)

    env._moc_cache[key] = nearest
    return nearest


def get_tcp_pos_w(env: "ManagerBasedRLEnv", ee_frame_name: str = "ee_frame") -> torch.Tensor:
    _invalidate_cache_if_needed(env)
    key = f"tcp_pos_w:{ee_frame_name}"
    if key in env._moc_cache:
        return env._moc_cache[key]

    tf_pos = env.scene[ee_frame_name].data.target_pos_w
    if tf_pos.ndim == 3 and tf_pos.shape[1] >= 2:
        pos = 0.5 * (tf_pos[:, 0, :] + tf_pos[:, 1, :])
    else:
        pos = tf_pos[:, 0, :]

    env._moc_cache[key] = pos
    return pos


def _quat_hemisphere_align(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(q * q_ref, dim=-1, keepdim=True)
    return torch.where(dot < 0.0, -q, q)


def get_tcp_quat_w(
    env: "ManagerBasedRLEnv",
    ee_frame_name: str = "ee_frame",
    mode: str = "avg",
) -> torch.Tensor:
    _invalidate_cache_if_needed(env)
    key = f"tcp_quat_w:{ee_frame_name}:{mode}"
    if key in env._moc_cache:
        return env._moc_cache[key]

    tf_quat = env.scene[ee_frame_name].data.target_quat_w
    if tf_quat.ndim == 3 and tf_quat.shape[1] >= 2:
        q0 = tf_quat[:, 0, :]
        q1 = tf_quat[:, 1, :]

        if mode == "left":
            quat = q0
        elif mode == "right":
            quat = q1
        else:
            q1 = _quat_hemisphere_align(q1, q0)
            quat = q0 + q1
            quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp(min=1e-9)
    else:
        quat = tf_quat[:, 0, :]

    env._moc_cache[key] = quat
    return quat


def get_tcp_pose_w(
    env: "ManagerBasedRLEnv",
    ee_frame_name: str = "ee_frame",
    quat_mode: str = "avg",
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_tcp_pos_w(env, ee_frame_name), get_tcp_quat_w(env, ee_frame_name, mode=quat_mode)