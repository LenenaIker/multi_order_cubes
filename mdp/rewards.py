from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

from .events import next_trigger_mask
from .step_cache import get_active_cube_pos_w, get_slots_w, get_tcp_pos_w, get_tcp_quat_w

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
    k_z: float = 0.06,
    p: float = 1.0,
    gate_dxy: float = 0.18,
    gate_band: float = 0.05,
) -> torch.Tensor:
    """Rational falloff (same family as `reward_reach_xy_rational`), not a Gaussian.

    A Gaussian bump with a tight sigma has ~zero gradient once dz is more than a few sigma
    away, so an agent starting far from the target gets no directional signal to close in
    (confirmed: dz sat flat at ~0.45m for a full 3M-step run with sigma_z=0.06, ~7.5 sigma
    away, reward numerically underflowing to 0). The rational form has fat tails and a
    nonzero gradient at any distance, so it can actually shape the initial descent.

    Target is the cube's own center (no extra standoff): `ee_frame`'s body_offset already
    places the TCP at the intended grasp point (see config/ur10_gripper/moc_ur10_env_cfg.py).
    No orientation reward is currently active (removed, see git history), so this target
    doesn't assume any particular gripper orientation.
    """
    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    cube = _target_cube_pos_w(env)

    dist_xy = _safe_norm(tip[:, :2] - cube[:, :2])

    dz = tip[:, 2] - cube[:, 2]
    k = float(max(1e-6, k_z))
    pp = float(max(1e-3, p))
    z_reward = 1.0 / (1.0 + torch.pow(torch.abs(dz) / k, pp))

    gate = torch.sigmoid((float(gate_dxy) - dist_xy) / float(max(1e-6, gate_band)))

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/reach_gate_xy"] = gate
    env.extras["moc/reach_abs_dz"] = torch.abs(dz)

    return gate * z_reward


def penalty_table_proximity(
    env: "ManagerBasedRLEnv",
    safe_height: float = 0.015,
    weight_per_m: float = 20.0,
    max_excess: float = 0.03,
) -> torch.Tensor:
    """Soft, continuous penalty for the TCP dropping too close to the table plane.

    Only fires below `safe_height` (world height above this env's local table plane, the
    table's top sits at `env_origins`'s own z), which is well below `reach_z_gated`'s own
    optimum (~0.03m, the target cube's center, see `config/ur10_gripper/moc_ur10_env_cfg.py`
    for the cube spawn height), so this doesn't fight that already-validated target, it only
    discourages overshooting past it toward the table.

    Added after visually observing the gripper and arm making much more contact with the
    table once a (now-removed, see git history) vertical top-down orientation reward started
    actually working: pointing the gripper's local +Z straight down drove the real fingertips,
    which extend a few cm past the TCP reference along that same axis, into the table, since
    the TCP target sits at the cube's center (~3cm above the table). Currently unused (see the
    commented-out `table_proximity` term in `moc_env_cfg.py`): no orientation reward is active
    anymore, so there's no longer an orientation-driven reason for the fingertips to dip below
    the TCP target on approach. Left here in case that changes again. Neither table contact nor
    arm self-collision currently have any physical consequence (`enabled_self_collisions=False`,
    `activate_contact_sensors=False` on UR10e_ROBOTIQ_GRIPPER_CFG), so nothing previously
    discouraged it. `max_excess` caps the penalty the same way `penalty_bystander_displacement`
    does, so a single-step depenetration spike near contact can't produce an unbounded per-step
    value.
    """
    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    height = tip[:, 2] - env.scene.env_origins[:, 2]

    excess = torch.clamp(float(safe_height) - height, min=0.0, max=float(max_excess))
    penalty = -float(weight_per_m) * excess

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/table_proximity_excess"] = excess

    return penalty


def penalty_arm_joint_velocity(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    joint_names: list[str] | None = None,
    max_l2: float = 50.0,
) -> torch.Tensor:
    """Continuous L2 penalty on arm joint velocity, meant to damp tremor.

    `max_l2` caps the raw penalty before it's scaled by the term's weight, the same pattern
    used by `penalty_bystander_displacement`'s `max_excess`. Without a cap, a single-step
    physics glitch (arm-cube contact blow-up) can spike joint velocity to nonphysical values,
    and squaring it produces an unbounded reward magnitude that poisons the SAC critic (see
    moc/reward_wiring_status for the earlier incident this exact pattern caused with the
    bystander-displacement term).
    """
    robot = env.scene[asset_name]

    joint_ids, _ = robot.find_joints(joint_names)

    qd = robot.data.joint_vel[:, joint_ids]

    penalty = torch.clamp(torch.sum(qd * qd, dim=-1), max=float(max_l2))

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/arm_joint_vel_l2"] = penalty

    return penalty


def reward_next_signal(
    env: "ManagerBasedRLEnv",
    next_threshold: float = 0.5,
    cooldown_steps: int = 30,
    success_xy: float = 0.05,
    success_z: float = 0.03,
    bonus: float = 5.0,
    penalty: float = -0.03,
) -> torch.Tensor:
    """Rewards pressing NEXT once the reach target is actually reached, penalizes early presses.

    The bonus is one-shot, tied to `next_trigger_mask` (the same condition that actually makes
    `consume_next_signal` resample the command), so it only fires on the exact step the command
    changes. The penalty is deliberately NOT trigger-gated: it applies every step the policy
    holds NEXT above threshold while not in the success zone, regardless of cooldown. A policy
    has no reason to release a continuous action once it discovers holding it is good, and an
    edge/cooldown-gated penalty would only ever catch the first press after an idle period,
    leaving every later resample during a sustained hold un-penalized. Because this fires every
    such step, its magnitude must stay tiny relative to the per-step reach reward (weight <<
    reach_xy/reach_z), or it would dominate and teach the policy to never press NEXT at all.

    Also updates `env.moc_stable_success`, which doubles as the "am I in the success zone
    right now" observation consumed by `mdp.observations.stable_success_hint`.
    """
    dist_xy = env.extras.get("moc/reach_dist_xy") if hasattr(env, "extras") and env.extras else None
    abs_dz = env.extras.get("moc/reach_abs_dz") if hasattr(env, "extras") and env.extras else None

    if dist_xy is None or abs_dz is None:
        success = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    else:
        success = (dist_xy < float(success_xy)) & (abs_dz < float(success_z))

    env.moc_stable_success = success.to(torch.float32)

    if hasattr(env, "moc_next_signal") and env.moc_next_signal is not None:
        pressing = env.moc_next_signal > float(next_threshold)
    else:
        pressing = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    trigger = next_trigger_mask(env, next_threshold=next_threshold, cooldown_steps=cooldown_steps)

    reward = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
    reward = torch.where(pressing & ~success, torch.full_like(reward, float(penalty)), reward)
    reward = torch.where(trigger & success, torch.full_like(reward, float(bonus)), reward)

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/next_reward"] = reward

    return reward


def penalty_bystander_displacement(
    env: "ManagerBasedRLEnv",
    tolerance_xy: float = 0.02,
    weight_per_m: float = 5.0,
    max_excess: float = 0.5,
) -> torch.Tensor:
    """Soft, continuous penalty for disturbing cubes that are NOT the current target.

    Gives corrective gradient from the very first accidental bump (elbow or otherwise),
    instead of relying solely on the hard `cube_off_table` termination for that signal.

    `excess` is capped at `max_excess` so a single physics blow-up (a cube flung meters away
    by a bad contact) can't produce an unbounded per-step penalty. Uncapped, this term hit
    the hundreds-of-thousands range for the whole rest of an episode and blew up the SAC
    critic (see moc/reward_wiring_status). `cube_fell_off_table`'s xy-displacement check now
    also ends the episode when this happens, so the cap here is a one-step safety net, not
    the primary fix.
    """
    if not hasattr(env, "moc_cube_home_pos_w") or env.moc_cube_home_pos_w is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    cubes_now_xy = get_active_cube_pos_w(env)[:, :, :2]
    cubes_home_xy = env.moc_cube_home_pos_w[:, :, :2]
    disp = _safe_norm(cubes_now_xy - cubes_home_xy)

    target_id = env.target_cube_id.to(torch.long).clamp(0, disp.shape[1] - 1)
    is_target = torch.zeros_like(disp, dtype=torch.bool)
    is_target.scatter_(1, target_id.view(-1, 1), True)

    bystander_disp = torch.where(is_target, torch.zeros_like(disp), disp)
    excess = torch.clamp(bystander_disp - float(tolerance_xy), min=0.0, max=float(max_excess))
    penalty = -float(weight_per_m) * excess.sum(dim=1)

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/bystander_disp"] = bystander_disp.sum(dim=1)

    return penalty


def reward_object_lifted(
    env: "ManagerBasedRLEnv",
    target_height: float = 0.10,
    tolerance: float = 0.001,
) -> torch.Tensor:
    """Dense reward for lifting the target cube above its resting height.

    Ramps linearly 0 -> 1 as the target cube's height above its own recorded home Z (see
    `moc_cube_home_pos_w`, set once per episode in `mdp/events.py`) goes from `tolerance` to
    `target_height`, then saturates at 1.0. Chosen over a binary "is lifted" flag so there is
    gradient across the whole climb, not just at the instant the threshold is crossed.
    Saturating (rather than rewarding height unboundedly) mirrors the `max_excess` cap in
    `penalty_bystander_displacement`: a physics-glitch fling shouldn't out-earn a real lift.

    First step toward Grasp-Lift, deliberately no phase observation and no contact sensor
    (see project memory: Isaac Lab's own Lift task does the same with its `object_is_lifted`).
    """
    if not hasattr(env, "moc_cube_home_pos_w") or env.moc_cube_home_pos_w is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    delta_z = get_active_cube_pos_w(env)[:, :, 2] - env.moc_cube_home_pos_w[:, :, 2]
    target_id = env.target_cube_id.to(torch.long).clamp(0, delta_z.shape[1] - 1)
    target_delta_z = delta_z.gather(1, target_id.view(-1, 1)).squeeze(1)

    progress = (target_delta_z - float(tolerance)) / (float(target_height) - float(tolerance))
    reward = torch.clamp(progress, min=0.0, max=1.0)

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/lift_delta_z"] = target_delta_z

    return reward