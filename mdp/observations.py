from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .step_cache import (
    get_slots_w,
    get_active_cube_pos_w,
    get_active_cube_quat_w,
    get_nearest_slot_for_active_cubes_xy,
)

# -------------------------
# Slots + comando (from,to)
# -------------------------
def slot_positions_in_base_frame(env, robot_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    slots_w = get_slots_w(env)  # (N,4,3)

    # identity quat (1,4) -> (N,4,4) como view
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32)
    slots_quat_w = ident.view(1, 1, 4).expand(env.num_envs, 4, 4)

    slots_pos_base, _ = math_utils.subtract_frame_transforms(
        root_pos_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3),
        root_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),
        slots_w.reshape(-1, 3),
        slots_quat_w.reshape(-1, 4),
    )
    return slots_pos_base.view(env.num_envs, 4, 3).reshape(env.num_envs, 12)


def command_from_to_onehot(env, num_slots: int = 4):
    """One-hot encoding of (from_slot, to_slot). Safe during env init before reset buffers exist."""
    N = int(getattr(env, "num_envs", 1))
    device = getattr(env, "device", "cpu")

    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        # shape: (N, 2*num_slots) -> [from_onehot | to_onehot]
        return torch.zeros((N, 2 * int(num_slots)), dtype=torch.float32, device=device)

    cmd = env.command_from_to  # (N,2) in 1..num_slots
    from_idx = torch.clamp(cmd[:, 0] - 1, 0, num_slots - 1)
    to_idx = torch.clamp(cmd[:, 1] - 1, 0, num_slots - 1)

    from_oh = torch.zeros((N, num_slots), dtype=torch.float32, device=device)
    to_oh = torch.zeros((N, num_slots), dtype=torch.float32, device=device)
    from_oh.scatter_(1, from_idx.view(-1, 1), 1.0)
    to_oh.scatter_(1, to_idx.view(-1, 1), 1.0)

    return torch.cat([from_oh, to_oh], dim=-1)

CUBE_KEYS_9 = [
    "cube_light_s", "cube_light_m", "cube_light_l",
    "cube_flat_s",  "cube_flat_m",  "cube_flat_l",
    "cube_dark_s",  "cube_dark_m",  "cube_dark_l",
]

def _active_cube_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    assert hasattr(env, "active_cube_indices"), "env.active_cube_indices missing. Call reset event randomize_cubes_on_slots first."
    pos9 = torch.stack([env.scene[k].data.root_pos_w for k in CUBE_KEYS_9], dim=1)  # (N,9,3)
    idx = env.active_cube_indices  # (N,3)
    return pos9.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (N,3,3)

def cubes_slot_occupancy_onehot(env: ManagerBasedRLEnv, num_slots: int = 4) -> torch.Tensor:
    nearest = get_nearest_slot_for_active_cubes_xy(env, num_slots=num_slots)  # (N,3)
    occ = torch.zeros((env.num_envs, num_slots), dtype=torch.float32, device=env.device)
    occ.scatter_(1, nearest, 1.0)
    return occ
# -------------------------
# Robot + EE + cubos
# -------------------------

def ee_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """End-effector pose in robot base frame. Output (N,3) / (N,4) / (N,7)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    if return_key == "pos":
        return ee_pos_b
    if return_key == "quat":
        return ee_quat_b
    return torch.cat([ee_pos_b, ee_quat_b], dim=1)

def cubes_poses_in_base_frame(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    cubes_pos_w = get_active_cube_pos_w(env)   # (N,3,3)
    cubes_quat_w = get_active_cube_quat_w(env) # (N,3,4)

    # Vectorize: treat the 3 cubes as a batch of 3 frames per env
    N = env.num_envs
    root_pos_rep = root_pos_w.unsqueeze(1).expand(N, 3, 3).reshape(-1, 3)
    root_quat_rep = root_quat_w.unsqueeze(1).expand(N, 3, 4).reshape(-1, 4)

    cubes_pos_flat = cubes_pos_w.reshape(-1, 3)
    cubes_quat_flat = cubes_quat_w.reshape(-1, 4)

    pos_b, quat_b = math_utils.subtract_frame_transforms(
        root_pos_rep, root_quat_rep, cubes_pos_flat, cubes_quat_flat
    )

    out = torch.cat([pos_b, quat_b], dim=1).view(N, 3, 7)  # (N,3,7)
    return out.reshape(N, 21)  # (N,21)


def gripper_state(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Robotiq 1-DOF gripper observation:
      - finger_joint position (N,1)
    """
    robot: Articulation = env.scene[robot_cfg.name]

    joint_names = getattr(env.cfg, "gripper_joint_names", ["finger_joint"])
    joint_ids, _ = robot.find_joints(joint_names)

    finger_q = robot.data.joint_pos[:, joint_ids[0]].to(torch.float32).unsqueeze(1)
    return finger_q


def stable_success_hint(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    (N,1) float in {0,1}. This is optional but stabilizes NEXT learning.
    Requires env.moc_stable_success to exist; if not, returns zeros.
    """
    if hasattr(env, "moc_stable_success") and env.moc_stable_success is not None:
        return env.moc_stable_success.to(torch.float32).view(-1, 1)
    return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)


def next_cooldown_obs(env, max_cooldown_steps: int = 30) -> torch.Tensor:
    """Expose NEXT cooldown as [0,1]. Helps Markovianity if cooldown affects rewards."""
    if not hasattr(env, "moc_next_cooldown") or env.moc_next_cooldown is None:
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

    cd = env.moc_next_cooldown.to(torch.float32).clamp(min=0.0)
    denom = float(max(1, int(max_cooldown_steps)))
    return (cd / denom).unsqueeze(1)

def moc_phase_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    (N,1) phase normalized to [0,1]. If missing, returns 0.
    """
    if hasattr(env, "moc_phase") and env.moc_phase is not None:
        # phases 1..5 -> scale to 0..1
        p = env.moc_phase.to(torch.float32).clamp(1, 5)
        return ((p - 1.0) / 4.0).view(-1, 1)
    return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)



# -------------------------
# Observación final "policy"
# -------------------------
def policy_obs(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    """
    Observation vector (recommended):

    - cubes poses in base frame:                 21
    - EE pose in base frame:                      7
    - slot positions in base frame (4*3):        12
    - occupancy mask of slots (4):                4
    - command from/to one-hot (4+4):              8
    - gripper state:                              1 (suction) or 2 (parallel)

    Total: 52 (+1 if suction, +2 if parallel)
    """
    obs = [
        cubes_poses_in_base_frame(env, robot_cfg=robot_cfg),
        ee_pose_in_base_frame(env, ee_frame_cfg, robot_cfg, return_key=None),
        slot_positions_in_base_frame(env, robot_cfg),
        cubes_slot_occupancy_onehot(env, num_slots=4),
        command_from_to_onehot(env, num_slots=4),
        gripper_state(env, robot_cfg),
        stable_success_hint(env),
        moc_phase_obs(env),
        next_cooldown_obs(env),
    ]
    return torch.cat(obs, dim=1)

