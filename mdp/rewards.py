from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .terminations import move_success
from .commands import ensure_command_buffer, ensure_moc_buffers
from .step_cache import get_active_cube_pos_w, get_slots_w, get_tcp_pos_w


# -----------------------------------------------------------------------------
# Basic geometry helpers
# -----------------------------------------------------------------------------

def _arange_env(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device)



def get_target_cube_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,3) Target cube world position (active cube indexed by env.target_cube_id)."""
    cubes = get_active_cube_pos_w(env)  # (N,3,3)
    idx = _arange_env(env)

    tid = None
    if hasattr(env, "target_cube_id") and env.target_cube_id is not None:
        tid = env.target_cube_id
    elif hasattr(env, "moc_target_cube_id") and env.moc_target_cube_id is not None:
        tid = env.moc_target_cube_id

    if tid is None:
        tid = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

    tid = tid.to(torch.long).clamp(0, cubes.shape[1] - 1)
    return cubes[idx, tid, :]


def get_to_slot_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,3) 'to' slot world position (command is stored as 1..4)."""
    slots = get_slots_w(env)  # (N,4,3)
    idx = _arange_env(env)

    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        return slots[idx, torch.zeros((env.num_envs,), dtype=torch.long, device=env.device), :]

    cmd = env.command_from_to.to(device=env.device)
    to_idx = torch.clamp(cmd[:, 1] - 1, 0, slots.shape[1] - 1).to(torch.long)
    return slots[idx, to_idx, :]


def _table_z(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,) approx table height using slot z (assumed constant across slots)."""
    slots = get_slots_w(env)
    return slots[:, 0, 2]

def _sigmoid(x: torch.Tensor, k: float = 10.0) -> torch.Tensor:
    return torch.sigmoid(k * x)

def _grasp_zone_mask(
    env: "ManagerBasedRLEnv",
    radius_xy: float = 0.05,
    z_lo: float = -0.02,
    z_hi: float = 0.05,
) -> torch.Tensor:
    """
    (N,) float mask 0..1 indicando si el tip está en una zona razonable de pregrasp.
    """
    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")  # (N,3)          # (N,3)
    cube = get_target_cube_pos_w(env) # (N,3)

    dxy = tip[:, :2] - cube[:, :2]
    dist_xy = torch.sqrt((dxy * dxy).sum(dim=-1) + 1e-12)
    dz = tip[:, 2] - cube[:, 2]

    in_xy = (dist_xy <= float(radius_xy))
    in_z = (dz >= float(z_lo)) & (dz <= float(z_hi))
    return (in_xy & in_z).to(torch.float32)


# -----------------------------------------------------------------------------
# Suction helpers (command vs. state)
# -----------------------------------------------------------------------------
# --- INSERT in mdp/rewards.py ---
def _get_finger_q(env: "ManagerBasedRLEnv", robot_name: str = "robot") -> torch.Tensor:
    """(N,) finger_joint position."""
    robot = env.scene[robot_name]
    joint_names = getattr(env.cfg, "gripper_joint_names", ["finger_joint"])
    joint_ids, _ = robot.find_joints(joint_names)
    return robot.data.joint_pos[:, joint_ids[0]].to(torch.float32)


def _get_gripper_close_cmd(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    (N,) proxy de 'intención de cerrar' desde el action term "gripper".
    Soporta distintos nombres de buffers según versión/acción.
    Devuelve float32 en device correcto.
    """
    z = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    if not hasattr(env, "action_manager"):
        return z

    try:
        term = env.action_manager.get_term("gripper")

        # Prioridad típica: processed_actions -> raw_actions -> action
        a = None
        for attr in ("processed_actions", "raw_actions", "action"):
            if hasattr(term, attr):
                a = getattr(term, attr)
                if a is not None:
                    break

        if a is None:
            return z

        a = a.to(device=env.device, dtype=torch.float32)

        # Esperamos 1-DOF -> (N,1) o (N,)
        if a.ndim == 2:
            return a[:, 0]
        return a
    except Exception:
        return z


def _get_gripper_closed(env: "ManagerBasedRLEnv", close_thr: float = 0.20) -> torch.Tensor:
    """
    (N,) boolean proxy 'gripper closed' from joint position.
    close_thr MUST be tuned to your Robotiq joint range.
    """
    return (_get_finger_q(env) > float(close_thr))




# -----------------------------------------------------------------------------
# NEXT helpers (edge detection + cooldown + stable success)
# -----------------------------------------------------------------------------

def _reset_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,) bool - envs that just reset this step (best-effort)."""
    if hasattr(env, "reset_buf") and env.reset_buf is not None:
        return env.reset_buf.to(torch.bool)
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        return (env.episode_length_buf == 0)
    return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)


def _ensure_next_buffers(env: "ManagerBasedRLEnv") -> None:
    ensure_moc_buffers(env)  # ensures moc_next_prev exists
    if not hasattr(env, "moc_next_cooldown") or env.moc_next_cooldown is None:
        env.moc_next_cooldown = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)
    if not hasattr(env, "moc_success_streak") or env.moc_success_streak is None:
        env.moc_success_streak = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)
    if not hasattr(env, "moc_stable_success") or env.moc_stable_success is None:
        env.moc_stable_success = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)


def _next_fired(env: "ManagerBasedRLEnv", tau: float = 0.0) -> torch.Tensor:
    """(N,) bool rising-edge on moc_next_signal > tau."""
    _ensure_next_buffers(env)
    sig = getattr(env, "moc_next_signal", None)
    if sig is None:
        cur = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    else:
        cur = (sig.view(-1) > float(tau))

    prev = env.moc_next_prev.to(torch.bool)
    fired = cur & (~prev)
    env.moc_next_prev = cur

    # reset boundary: clear prev so first step doesn't create a false edge
    rm = _reset_mask(env)
    if rm.any():
        env.moc_next_prev = torch.where(rm, torch.zeros_like(env.moc_next_prev), env.moc_next_prev)

    return fired


def _update_stable_success(
    env: "ManagerBasedRLEnv",
    stable_window: int = 5,
    tol_xy: float = 0.025,
    tol_z: float = 0.05,
    require_to_clear: bool = True,
    require_settled: bool = True,
    vel_tol: float = 0.20,
) -> torch.Tensor:
    """
    Updates env.moc_success_streak and env.moc_stable_success.
    stable_success becomes True after `stable_window` consecutive success steps.
    """
    _ensure_next_buffers(env)

    suc = move_success(
        env,
        tol_xy=tol_xy,
        tol_z=tol_z,
        require_to_clear=require_to_clear,
        require_settled=require_settled,
        vel_tol=vel_tol,
    )

    streak = env.moc_success_streak
    streak = torch.where(suc, streak + 1, torch.zeros_like(streak))
    env.moc_success_streak = streak

    stable = streak >= int(max(1, stable_window))
    env.moc_stable_success = stable

    rm = _reset_mask(env)
    if rm.any():
        env.moc_success_streak = torch.where(rm, torch.zeros_like(env.moc_success_streak), env.moc_success_streak)
        env.moc_stable_success = torch.where(rm, torch.zeros_like(env.moc_stable_success), env.moc_stable_success)

    return stable


def _cooldown_tick(env: "ManagerBasedRLEnv") -> None:
    _ensure_next_buffers(env)
    cd = env.moc_next_cooldown
    cd = torch.clamp(cd - 1, min=0)
    env.moc_next_cooldown = cd

    rm = _reset_mask(env)
    if rm.any():
        env.moc_next_cooldown = torch.where(rm, torch.zeros_like(env.moc_next_cooldown), env.moc_next_cooldown)


# -----------------------------------------------------------------------------
# Reward terms (new, minimal, stable)
# -----------------------------------------------------------------------------

def reward_tip_to_target_xy(env, sigma_xy: float = 0.25) -> torch.Tensor:
    ensure_command_buffer(env)

    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")  # (N,3)                 # (N,3)
    cube = get_target_cube_pos_w(env)       # (N,3)

    dxy = tip[:, :2] - cube[:, :2]
    d2 = (dxy * dxy).sum(dim=-1)
    s2 = float(max(1e-6, sigma_xy)) ** 2
    r = torch.exp(-0.5 * d2 / s2)

    # logging
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/d_tip_cube_xy"] = torch.sqrt(d2 + 1e-8)
    env.extras["moc/tip_xy_x"] = tip[:, 0]
    env.extras["moc/tip_xy_y"] = tip[:, 1]
    env.extras["moc/cube_xy_x"] = cube[:, 0]
    env.extras["moc/cube_xy_y"] = cube[:, 1]
    ee = env.scene["ee_frame"]
    
    
    ee = env.scene["ee_frame"]
    names = getattr(ee.data, "target_frame_names", None)
    if names is not None and (not hasattr(env, "_printed_ee_names")):
        print("EE target_frame_names:", names)
        env._printed_ee_names = True

    if not hasattr(env, "_printed_ee_shape"):
        print("EE target_pos_w shape:", ee.data.target_pos_w.shape)
        env._printed_ee_shape = True
    

    return r


def reward_tip_to_target_z(env, z_offset: float = 0.0, sigma_z: float = 0.05) -> torch.Tensor:
    """
    Reward denso para acercar el tip en Z al cubo target (siempre, sin gating).
    z_offset: si quieres apuntar a 'pregrasp' encima del cubo (p.ej. 0.10), ponlo aquí.
    sigma_z: ancho del shaping (más pequeño = gradiente más fuerte cerca).
    Tiene que tener un weight inferior a XY para que no sea vago y aprenda solo esto
    """
    ensure_command_buffer(env)

    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")  # (N,3)           # (N,3)
    cube = get_target_cube_pos_w(env) # (N,3)

    dz = tip[:, 2] - (cube[:, 2] + float(z_offset))
    d2 = dz * dz
    s2 = float(max(1e-6, sigma_z)) ** 2
    r = torch.exp(-0.5 * d2 / s2)

    # logging
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/d_tip_cube_z"] = torch.sqrt(d2 + 1e-8)
    env.extras["moc/tip_z"] = tip[:, 2]
    env.extras["moc/cube_z"] = cube[:, 2]

    return r


def reward_lift_target_z(
    env: "ManagerBasedRLEnv",
    z_lift_min: float = 0.02,
    z_lift_max: float = 0.12,
    grasp_radius_xy: float = 0.05,
    grasp_z_lo: float = -0.02,
    grasp_z_hi: float = 0.05,
    close_k: float = 6.0,
) -> torch.Tensor:
    """
    Lift shaping con gating suave:
      lift = clamp((lift_h - z_min)/(z_max-z_min), 0,1)
      gate = in_grasp_zone * sigmoid(close_cmd, k)
      r = gate * lift
    """
    ensure_command_buffer(env)  # sin latch por step

    cube = get_target_cube_pos_w(env)
    lift_h = cube[:, 2] - _table_z(env)

    den = float(max(1e-6, z_lift_max - z_lift_min))
    lift = torch.clamp((lift_h - float(z_lift_min)) / den, 0.0, 1.0)

    in_zone = _grasp_zone_mask(env, radius_xy=grasp_radius_xy, z_lo=grasp_z_lo, z_hi=grasp_z_hi)
    close_cmd = _get_gripper_close_cmd(env)  # (N,)
    gate = in_zone * _sigmoid(close_cmd, k=close_k)

    return gate * lift



def reward_move_target_to_goal_xy(
    env: "ManagerBasedRLEnv",
    sigma_xy: float = 0.12,
    z_lift_gate: float = 0.02,
    z_lift_band: float = 0.06,
    close_k: float = 6.0,
) -> torch.Tensor:
    """
    Transport shaping con gate suave:
      base = exp(-||cube_xy - goal_xy||^2 / (2*sigma^2))
      lifted = clamp((lift_h - z_gate)/band, 0,1)
      gate = sigmoid(close_cmd,k) * lifted
      r = gate * base
    """
    ensure_command_buffer(env)

    cube = get_target_cube_pos_w(env)
    goal = get_to_slot_pos_w(env)

    dxy = cube[:, :2] - goal[:, :2]
    d2 = (dxy * dxy).sum(dim=-1)
    s2 = float(max(1e-6, sigma_xy)) ** 2
    base = torch.exp(-0.5 * d2 / s2)

    lift_h = cube[:, 2] - _table_z(env)
    lifted = torch.clamp((lift_h - float(z_lift_gate)) / float(max(1e-6, z_lift_band)), 0.0, 1.0)

    close_cmd = _get_gripper_close_cmd(env)
    gate = _sigmoid(close_cmd, k=close_k) * lifted

    return gate * base


def reward_close_cmd_in_grasp_zone(
    env: "ManagerBasedRLEnv",
    radius_xy: float = 0.04,
    z_lo: float = -0.01,
    z_hi: float = 0.03,
) -> torch.Tensor:
    """
    Equivalent to old suction_cmd_in_zone:
      reward if the *command* is closing while EE is in grasp zone over cube.
    """
    ensure_command_buffer(env)

    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")  # (N,3)
    cube = get_target_cube_pos_w(env)

    dxy = tip[:, :2] - cube[:, :2]
    in_xy = (dxy * dxy).sum(dim=-1) <= float(radius_xy) ** 2

    dz = tip[:, 2] - cube[:, 2]
    in_z = (dz >= float(z_lo)) & (dz <= float(z_hi))

    in_zone = in_xy & in_z

    close_cmd = _get_gripper_close_cmd(env)
    # interpret "closing intent": higher command -> more closed (depends on your action convention)
    # if your JointPositionAction is absolute target, then use a threshold:
    closing = (close_cmd > 0.0)
    
    return (in_zone & closing).to(torch.float32)


# --- REPLACE in mdp/rewards.py ---
def penalty_close_cmd_outside_grasp_zone(
    env: "ManagerBasedRLEnv",
    radius_xy: float = 0.06,
    z_lo: float = -0.02,
    z_hi: float = 0.05,
) -> torch.Tensor:
    """
    Penalize closing intent when not in grasp zone.
    """
    ensure_command_buffer(env)

    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")  # (N,3)
    cube = get_target_cube_pos_w(env)

    dxy = tip[:, :2] - cube[:, :2]
    in_xy = (dxy * dxy).sum(dim=-1) <= float(radius_xy) ** 2

    dz = tip[:, 2] - cube[:, 2]
    in_z = (dz >= float(z_lo)) & (dz <= float(z_hi))

    in_zone = in_xy & in_z

    close_cmd = _get_gripper_close_cmd(env)
    closing = (close_cmd > 0.0)
    return (closing & (~in_zone)).to(torch.float32)


def reward_next_commit(
    env: "ManagerBasedRLEnv",
    tau: float = 0.0,
    stable_window: int = 5,
    cooldown_steps: int = 30,
    R_next_ok: float = 10.0,
    R_next_bad: float = 1.0,
    R_wait: float = 0.05,
    tol_xy: float = 0.025,
    tol_z: float = 0.05,
    require_to_clear: bool = True,
    require_settled: bool = True,
    vel_tol: float = 0.20,
) -> torch.Tensor:
    """
    NEXT "commit" reward:
      +R_next_ok if NEXT fires (edge) and stable_success is True and cooldown==0.
      -R_next_bad if NEXT fires when not ready (and cooldown==0).
      -R_wait per step while ready but not committing (optional, small).
    """
    ensure_command_buffer(env)
    _ensure_next_buffers(env)

    # tick cooldown every step
    _cooldown_tick(env)
    cd_ok = env.moc_next_cooldown <= 0

    stable = _update_stable_success(
        env,
        stable_window=stable_window,
        tol_xy=tol_xy,
        tol_z=tol_z,
        require_to_clear=require_to_clear,
        require_settled=require_settled,
        vel_tol=vel_tol,
    )

    fired = _next_fired(env, tau=tau)
    can_fire = fired & cd_ok

    ok = can_fire & stable
    bad = can_fire & (~stable)

    # apply cooldown on any NEXT attempt that we accepted (can_fire)
    if can_fire.any():
        env.moc_next_cooldown = torch.where(
            can_fire, torch.full_like(env.moc_next_cooldown, int(cooldown_steps)), env.moc_next_cooldown
        )

    rew = ok.to(torch.float32) * float(R_next_ok) - bad.to(torch.float32) * float(R_next_bad)

    # "linger" penalty after success to push for NEXT
    rew = rew - stable.to(torch.float32) * (1.0 - fired.to(torch.float32)) * float(R_wait)

    # debug flags (optional)
    env.moc_commit_ok = ok.to(torch.bool)
    env.moc_commit_bad = bad.to(torch.bool)

    return rew


# -----------------------------------------------------------------------------
# Optional standard penalty (kept, low weight)
# -----------------------------------------------------------------------------

def reward_penalty_joint_vel(env: "ManagerBasedRLEnv", lambda_vel: float = 0.01) -> torch.Tensor:
    """Small L2 penalty on joint velocities (helps smoothness)."""
    robot = env.scene["robot"]
    qd = robot.data.joint_vel
    return -float(lambda_vel) * torch.sum(qd * qd, dim=-1)







def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1) + eps)


def _smooth_gate(d: torch.Tensor, d_on: float, band: float) -> torch.Tensor:
    """
    Smooth gate ~1 when d <= d_on, ~0 when d >> d_on.
    Implemented as sigmoid((d_on - d)/band).
    """
    b = float(max(1e-6, band))
    return torch.sigmoid((float(d_on) - d) / b)

def reward_reach_xy_rational(
    env: "ManagerBasedRLEnv",
    k_xy: float = 0.10,     # “radio” de caída (m)
    p: float = 1.0,         # potencia opcional
) -> torch.Tensor:
    """
    r = 1 / (1 + (dxy/k)^p)
    - No satura tan rápido como exp(-d^2)
    - Gradiente útil lejos
    """
    ensure_command_buffer(env)
    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    cube = get_target_cube_pos_w(env)

    dxy = tip[:, :2] - cube[:, :2]
    dist_xy = _safe_norm(dxy)

    k = float(max(1e-6, k_xy))
    pp = float(max(1e-3, p))
    r = 1.0 / (1.0 + torch.pow(dist_xy / k, pp))

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/reach_dist_xy"] = dist_xy
    return r


def reward_reach_xy_progress(
    env: "ManagerBasedRLEnv",
    scale: float = 1.0,     # ganancia del progreso
    clip: float = 0.02,     # clip en metros por step (evita explosiones)
) -> torch.Tensor:
    """
    r = scale * clip(d_prev - d_now, [-clip, clip])
    - quieto => ~0
    - acercarse => positivo
    - alejarse => negativo
    """
    ensure_command_buffer(env)
    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    cube = get_target_cube_pos_w(env)

    dxy = tip[:, :2] - cube[:, :2]
    dist_xy = _safe_norm(dxy)

    # init buffers
    if not hasattr(env, "_moc_prev_dist_xy") or env._moc_prev_dist_xy is None:
        env._moc_prev_dist_xy = dist_xy.detach()

    prev = env._moc_prev_dist_xy
    # si hay reset_buf, igualamos prev=dist para no dar reward artificial
    if hasattr(env, "reset_buf") and env.reset_buf is not None:
        rb = env.reset_buf.to(dtype=torch.bool)
        prev = torch.where(rb, dist_xy.detach(), prev)

    delta = (prev - dist_xy)  # positivo si se acerca
    c = float(max(1e-6, clip))
    delta = torch.clamp(delta, -c, c)

    # update prev
    env._moc_prev_dist_xy = dist_xy.detach()

    r = float(scale) * delta

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/reach_delta_xy"] = r
    return r

def reward_reach_z_gated(
    env: "ManagerBasedRLEnv",
    z_offset: float = 0.10,
    sigma_z: float = 0.06,
    gate_dxy: float = 0.18,     # <- más grande que antes
    gate_band: float = 0.05,    # <- más suave
) -> torch.Tensor:
    ensure_command_buffer(env)
    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    cube = get_target_cube_pos_w(env)

    dxy = tip[:, :2] - cube[:, :2]
    dist_xy = _safe_norm(dxy)

    dz = tip[:, 2] - (cube[:, 2] + float(z_offset))
    s = float(max(1e-6, sigma_z))
    z_rew = torch.exp(-0.5 * (dz * dz) / (s * s))

    b = float(max(1e-6, gate_band))
    gate = torch.sigmoid((float(gate_dxy) - dist_xy) / b)

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras["moc/reach_gate_xy"] = gate
    env.extras["moc/reach_abs_dz"] = torch.abs(dz)

    return gate * z_rew



def reward_reach_success_bonus(
    env: "ManagerBasedRLEnv",
    tol_xy: float = 0.03,
    tol_z: float = 0.04,
    z_offset: float = 0.10,
    bonus: float = 1.0,
) -> torch.Tensor:
    """
    Sparse-ish but still smooth/controlled: give a bonus when inside a tight box.
    Helps SAC “lock onto” success and discourages camping elsewhere.

    Output in {0, bonus}.
    """
    ensure_command_buffer(env)

    tip = get_tcp_pos_w(env, ee_frame_name="ee_frame")
    cube = get_target_cube_pos_w(env)

    dxy = tip[:, :2] - cube[:, :2]
    dist_xy = _safe_norm(dxy)
    dz = tip[:, 2] - (cube[:, 2] + float(z_offset))

    ok = (dist_xy <= float(tol_xy)) & (torch.abs(dz) <= float(tol_z))
    return ok.to(torch.float32) * float(bonus)