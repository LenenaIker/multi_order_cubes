from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

from .events import next_trigger_mask
from .step_cache import (
    get_active_cube_pos_w,
    get_finger_contact_force_vec_w,
    get_finger_contact_force_w,
    get_slots_w,
    get_tcp_pos_w,
    get_tcp_quat_w,
)

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


def _log_weighted_reward(env: "ManagerBasedRLEnv", name: str, raw: torch.Tensor, weight: float) -> None:
    """Logs a reward term's actual contribution to the total reward SAC receives.

    `RewardManager.compute()` (isaaclab.managers.reward_manager) multiplies each term's raw
    return by `weight` and `env.step_dt` when summing the episode total, so that's what's
    logged here too, under a `rewards/` prefix `IsaacInfoTensorboardCallback` forwards to its
    own TensorBoard tag group. Lets us see which term the policy is actually maximizing in
    absolute terms, not just each function's raw (pre-weight) output shape.
    """
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras[f"rewards/{name}"] = raw.detach() * float(weight) * env.step_dt


def _finger_cos_sim(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Cosine similarity between the left/right finger contact-force vectors.

    Confirmed live via ContactSensorInspector.py (2026-08-30): a real cube pinch reads
    ~-0.97 to -1.00 (fingers pushed apart from each other), both fingers pressing flat
    against the table reads ~+1.00 (pushed the same direction). Used by
    reward_grasp_contact to reject the table-press false positive.
    """
    vecs = get_finger_contact_force_vec_w(env)
    left, right = vecs[:, 0, :], vecs[:, 1, :]
    left_n = _safe_norm(left)
    right_n = _safe_norm(right)
    cos = (left * right).sum(dim=-1) / (left_n * right_n).clamp(min=1e-6)
    # near-zero force on either finger means there's no real contact signal to read a
    # direction from -- don't let a near-zero-norm division artifact pass as a valid pinch.
    valid = (left_n > 1e-3) & (right_n > 1e-3)
    return torch.where(valid, cos, torch.zeros_like(cos))


def reward_reach_xy_rational(
    env: "ManagerBasedRLEnv",
    k_xy: float = 0.10,
    p: float = 1.0,
    weight: float = 12.0,
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
    _log_weighted_reward(env, "reach_xy", reward, weight)

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
    flat_margin: float = 0.03,
    weight: float = 9.5,
) -> torch.Tensor:
    """Rational falloff (same family as `reward_reach_xy_rational`), not a Gaussian.

    A Gaussian bump with a tight sigma has ~zero gradient once dz is more than a few sigma
    away, so an agent starting far from the target gets no directional signal to close in
    (confirmed: dz sat flat at ~0.45m for a full 3M-step run with sigma_z=0.06, ~7.5 sigma
    away, reward numerically underflowing to 0). The rational form has fat tails and a
    nonzero gradient at any distance, so it can actually shape the initial descent.

    `flat_margin` clips the falloff's input distance to 0 within that radius of the cube's
    own center, so the reward is exactly 1.0 anywhere inside it instead of peaking only at
    dz=0 (the cube's own center, i.e. inside its solid volume). Without this, the optimum
    sat inside the cube, giving a live gradient to push the TCP down into solid geometry
    with nothing rewarding a hover just above it. `flat_margin=0.03` was picked to clear the
    largest cube variant's half-height with margin. `k_z`/`p` are unchanged from before and
    still only control how the reward falls off beyond that flat radius, not its width.

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
    d_eff = torch.clamp(torch.abs(dz) - float(max(0.0, flat_margin)), min=0.0)
    z_reward = 1.0 / (1.0 + torch.pow(d_eff / k, pp))

    gate = torch.sigmoid((float(gate_dxy) - dist_xy) / float(max(1e-6, gate_band)))

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/reach_gate_xy"] = gate
    env.extras["moc/reach_abs_dz"] = torch.abs(dz)

    reward = gate * z_reward
    _log_weighted_reward(env, "reach_z", reward, weight)

    return reward


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
    weight: float = 1.0,
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
    _log_weighted_reward(env, "next_signal", reward, weight)

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


def diag_grip_distance(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    gripper_joint_name: str = "finger_joint",
    closed_threshold: float = 0.7,
) -> torch.Tensor:
    """Diagnostics only, register with weight=0.0 so it never affects training.

    Logs where the gripper is actually closing relative to the target cube: masks the
    reach distance/offset by which envs currently have their gripper past
    `closed_threshold` closedness (normalized against the joint's own live position
    limits, not a hardcoded angle), so `moc/dist_xy_at_grip` / `moc/dx_at_grip` /
    `moc/dy_at_grip` reflect only the moments the policy chooses to close, not the whole
    episode. `moc/grip_frac` (fraction of envs gripping this logged snapshot) is logged
    alongside it: when it's near 0, the other four values are near-meaningless (division
    by a clamped denominator when nobody's gripping falls back to 0, which would otherwise
    look like "gripping dead-center" rather than "nobody was gripping").

    Exists to check the suspicion that the policy learned to grip beside the cube rather
    than centered on it, without needing a new training run's reward shape to find out.
    """
    robot = env.scene[asset_name]
    joint_ids, _ = robot.find_joints([gripper_joint_name])
    jid = joint_ids[0]

    joint_pos = robot.data.joint_pos[:, jid]
    lower = robot.data.joint_pos_limits[:, jid, 0]
    upper = robot.data.joint_pos_limits[:, jid, 1]
    closed_frac = ((joint_pos - lower) / (upper - lower).clamp(min=1e-6)).clamp(0.0, 1.0)
    gripping = (closed_frac > float(closed_threshold)).to(torch.float32)
    n_gripping = gripping.sum().clamp(min=1.0)

    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    cube = _target_cube_pos_w(env)
    offset_xy = tip[:, :2] - cube[:, :2]
    dist_xy = _safe_norm(offset_xy)
    dist_z = torch.abs(tip[:, 2] - cube[:, 2])

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/grip_frac"] = gripping
    env.extras["moc/dist_xy_at_grip"] = ((dist_xy * gripping).sum() / n_gripping).expand(env.num_envs)
    env.extras["moc/dist_z_at_grip"] = ((dist_z * gripping).sum() / n_gripping).expand(env.num_envs)
    env.extras["moc/dx_at_grip"] = ((offset_xy[:, 0] * gripping).sum() / n_gripping).expand(env.num_envs)
    env.extras["moc/dy_at_grip"] = ((offset_xy[:, 1] * gripping).sum() / n_gripping).expand(env.num_envs)

    return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)


def reward_grasp_contact(
    env: "ManagerBasedRLEnv",
    success_xy: float = 0.05,
    success_z: float = 0.03,
    force_cap: float = 20.0,
    pinch_cos_threshold: float = -0.3,
    weight: float = 10.0,
) -> torch.Tensor:
    """Dense reward for squeezing the gripper closed on the target cube specifically.

    Uses `min(left_force, right_force)`, not sum or max, so a single finger brushing the
    cube's side or another object can't score alone; a real grasp has both fingers loaded.
    Gated by the same reach success zone `reward_next_signal` uses (TCP already within
    `success_xy`/`success_z` of the target cube), so contact against the table or a
    bystander cube during approach doesn't score, only squeezing while positioned on the
    actual target.

    Structurally safe from "squeeze fingers against each other with nothing in between":
    `enabled_self_collisions=False` on the gripper means closing on empty air reports
    exactly zero force on both sensors (confirmed live via ContactSensorInspector.py,
    2026-08-29), so nonzero force on both fingers can only come from a real object between
    them.

    Also gated on `_finger_cos_sim` being below `pinch_cos_threshold`: the contact sensors
    are unfiltered (report contact against anything, table included), and magnitude alone
    can't tell a real pinch apart from both fingers pressing flat against the table -- a
    20M-step run exploited exactly that, holding a stable fake grasp against the table
    instead of ever attempting a lift. `pinch_cos_threshold=-0.3` sits with a generous
    margin below the table-press value observed live (~+1.0) and above the real-pinch range
    (~-0.97 to -1.00), see project_contact_sensor_investigation.md memory.
    """
    dist_xy = env.extras.get("moc/reach_dist_xy") if hasattr(env, "extras") and env.extras else None
    abs_dz = env.extras.get("moc/reach_abs_dz") if hasattr(env, "extras") and env.extras else None

    if dist_xy is None or abs_dz is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    in_position = (dist_xy < float(success_xy)) & (abs_dz < float(success_z))

    force = get_finger_contact_force_w(env)
    grip_force = torch.amin(force, dim=1)
    cos_sim = _finger_cos_sim(env)
    pinch_gate = cos_sim < float(pinch_cos_threshold)
    reward = (
        (grip_force.clamp(min=0.0, max=float(force_cap)) / float(force_cap))
        * in_position.to(torch.float32)
        * pinch_gate.to(torch.float32)
    )

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/grasp_force"] = grip_force
    env.extras["moc/grasp_cos_sim"] = cos_sim
    _log_weighted_reward(env, "grasp_contact", reward, weight)

    return reward


def reward_object_lifted(
    env: "ManagerBasedRLEnv",
    target_height: float = 0.10,
    tolerance: float = 0.001,
    weight: float = 15.0,
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
    _log_weighted_reward(env, "object_lifted", reward, weight)

    return reward