from __future__ import annotations
import torch
from typing import TYPE_CHECKING

import re

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .terminations import move_success
from .commands import (
    sample_command_from_to,
    ensure_command_buffer,
    ensure_moc_buffers,
    latch_command_state,
)

from .step_cache import get_active_cube_pos_w, get_slots_w

# -------------------------
# Buffers / state helpers
# -------------------------

def _ensure_next_buffers(env: "ManagerBasedRLEnv") -> None:
    """Lazy init of NEXT-related buffers on env."""
    if not hasattr(env, "moc_success_count") or env.moc_success_count is None:
        env.moc_success_count = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    if not hasattr(env, "moc_stable_success") or env.moc_stable_success is None:
        env.moc_stable_success = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    if not hasattr(env, "moc_next_cooldown") or env.moc_next_cooldown is None:
        env.moc_next_cooldown = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    if not hasattr(env, "moc_next_signal") or env.moc_next_signal is None:
        # Written every step by your ActionTerm (NextFlagAction)
        env.moc_next_signal = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    # --- stamps to make updates idempotent per step (avoid 3x updates per reward-term) ---
    if not hasattr(env, "moc_last_success_update_step") or env.moc_last_success_update_step is None:
        env.moc_last_success_update_step = torch.full((env.num_envs,), -1, dtype=torch.int32, device=env.device)

    if not hasattr(env, "moc_last_cooldown_step") or env.moc_last_cooldown_step is None:
        env.moc_last_cooldown_step = torch.full((env.num_envs,), -1, dtype=torch.int32, device=env.device)

    if not hasattr(env, "moc_cmd_advanced_step") or env.moc_cmd_advanced_step is None:
        env.moc_cmd_advanced_step = torch.full((env.num_envs,), -1, dtype=torch.int32, device=env.device)





def _update_stable_success(
    env: "ManagerBasedRLEnv",
    stable_window: int,
) -> torch.Tensor:
    """Update moc_success_count & moc_stable_success ONCE per env-step (idempotent)."""
    _ensure_next_buffers(env)

    if not hasattr(env, "episode_length_buf"):
        raise AttributeError("env.episode_length_buf not found; needed to make stable_success update idempotent.")

    sid = env.episode_length_buf.to(torch.int32)  # (N,)
    needs_update = sid != env.moc_last_success_update_step

    ok = move_success(env)  # (N,) bool

    next_count = torch.where(ok, env.moc_success_count + 1, torch.zeros_like(env.moc_success_count))
    env.moc_success_count = torch.where(needs_update, next_count, env.moc_success_count)

    stable = env.moc_success_count >= int(stable_window)
    env.moc_stable_success = stable

    env.moc_last_success_update_step = torch.where(needs_update, sid, env.moc_last_success_update_step)
    return stable




def _next_fired(env: "ManagerBasedRLEnv", tau: float) -> torch.Tensor:
    """
    Convert continuous env.moc_next_signal into boolean NEXT fired.
    Assumes env.moc_next_signal in [-1,1] => tau=0 is typical.
    """
    _ensure_next_buffers(env)
    return env.moc_next_signal > float(tau)


def _cooldown_ok(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _ensure_next_buffers(env)
    return env.moc_next_cooldown <= 0


def _step_cooldown(env: "ManagerBasedRLEnv") -> None:
    """Decrement cooldown ONCE per env-step (idempotent)."""
    _ensure_next_buffers(env)

    if not hasattr(env, "episode_length_buf"):
        raise AttributeError("env.episode_length_buf not found; needed to make cooldown update idempotent.")

    sid = env.episode_length_buf.to(torch.int32)
    needs_update = sid != env.moc_last_cooldown_step

    dec = torch.clamp(env.moc_next_cooldown - 1, min=0)
    env.moc_next_cooldown = torch.where(needs_update, dec, env.moc_next_cooldown)

    env.moc_last_cooldown_step = torch.where(needs_update, sid, env.moc_last_cooldown_step)


def _trigger_cooldown(env: "ManagerBasedRLEnv", mask: torch.Tensor, cooldown_steps: int) -> None:
    """Set cooldown for envs where mask==True."""
    _ensure_next_buffers(env)
    cd = torch.full_like(env.moc_next_cooldown, int(cooldown_steps))
    env.moc_next_cooldown = torch.where(mask, cd, env.moc_next_cooldown)


# -------------------------
# NEXT rewards
# -------------------------

def reward_next_commit_success(
    env: "ManagerBasedRLEnv",
    tau: float = 0.0,
    stable_window: int = 3,
    cooldown_steps: int = 8,
    R_commit: float = 8.0,
    advance_command: bool = True,
) -> torch.Tensor:
    """
    +R_commit iff NEXT fired (and cooldown==0) AND stable_success==True.
    Optionally advances command (resamples from/to) only for commit-ok envs.
    """
    stable = _update_stable_success(env, stable_window)
    fired = _next_fired(env, tau)
    can_fire = _cooldown_ok(env)
    next_evt = fired & can_fire

    tool_success = _tool_success_mask(env, z_lift_min=0.03)  # stable_success & suction & lifted
    commit_ok = next_evt & tool_success


    _log_kv(env, "next_fired", fired.to(torch.float32))
    _log_kv(env, "stable_success", stable.to(torch.float32))
    _log_kv(env, "suction_on", (_get_suction_state(env) > 0).to(torch.float32))
    _log_kv(env, "lifted", _is_lifted_target(env, z_lift_min=0.03))
    _log_kv(env, "tool_success", tool_success.to(torch.float32))
    _log_kv(env, "commit_ok", commit_ok.to(torch.float32))


    env.moc_commit_ok = commit_ok.to(torch.bool)
    rew = commit_ok.to(torch.float32) * float(R_commit)

    # Apply cooldown on any NEXT attempt (ok or fail)
    _trigger_cooldown(env, next_evt, cooldown_steps)

    # Advance ONLY those envs that committed correctly
    if advance_command and commit_ok.any():
        # Advance at most once per env-step (protect against accidental multiple calls).
        sid = env.episode_length_buf.to(torch.int32) if hasattr(env, "episode_length_buf") else torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)
        can_advance = commit_ok & (env.moc_cmd_advanced_step != sid)
        if can_advance.any():
            env_ids = torch.nonzero(can_advance, as_tuple=False).squeeze(-1)
            sample_command_from_to(env, env_ids=env_ids)

            env.moc_cmd_advanced_step = torch.where(can_advance, sid, env.moc_cmd_advanced_step)

        # Reset stable counters for those envs (new subtask)
        zeros_i32 = torch.zeros_like(env.moc_success_count)
        zeros_b   = torch.zeros_like(env.moc_stable_success)
        env.moc_success_count = torch.where(commit_ok, zeros_i32, env.moc_success_count)
        env.moc_stable_success = torch.where(commit_ok, zeros_b, env.moc_stable_success)

    _step_cooldown(env)
    return rew


def reward_next_commit_fail(
    env: "ManagerBasedRLEnv",
    tau: float = 0.0,
    stable_window: int = 3,
    cooldown_steps: int = 8,
    R_false: float = 2.0,
) -> torch.Tensor:
    """
    -R_false iff NEXT fired (and cooldown==0) AND stable_success==False.
    """
    stable = _update_stable_success(env, stable_window)
    fired = _next_fired(env, tau)
    can_fire = _cooldown_ok(env)
    next_evt = fired & can_fire

    commit_bad = next_evt & (~stable)
    rew = -commit_bad.to(torch.float32) * float(R_false)

    # Apply cooldown on any NEXT attempt
    _trigger_cooldown(env, next_evt, cooldown_steps)
    _step_cooldown(env)
    return rew


def reward_wait_after_success(
    env: "ManagerBasedRLEnv",
    stable_window: int = 3,
    lambda_wait: float = 0.05,
) -> torch.Tensor:
    """
    Small negative reward when stable_success is True but agent does not fire NEXT.
    Prevents agent from stalling after completing the subtask.
    """
    stable = _update_stable_success(env, stable_window)
    fired = _next_fired(env, tau=0.0)  # threshold irrelevant here
    wait = stable & (~fired)
    return -wait.to(torch.float32) * float(lambda_wait)


# -------------------------
# Phase-1 stability shaping
# -------------------------
def reward_penalty_disturb_other_cubes(
    env: "ManagerBasedRLEnv",
    lambda_disturb: float = 0.25,
    tol_xy: float = 0.01,
) -> torch.Tensor:
    """Penalize moving non-target cubes away from their positions at command assignment.

    Requires buffers created by ensure_moc_buffers() and updated when commands are sampled:
        - env.target_cube_id: (N,) in {0,1,2}
        - env.moc_cmd_cube_pos_xy0: (N,3,2) baseline XY positions of active cubes
    """
    ensure_moc_buffers(env)

    if env.moc_cmd_stamp is None:
        env.moc_cmd_stamp = -torch.ones((env.num_envs,), dtype=torch.long, device=env.device)

    # On-demand initialization: if command exists but stamp is unset for some envs, latch baseline now.
    needs_latch = env.moc_cmd_stamp < 0
    if needs_latch.any() and hasattr(env, "command_from_to") and env.command_from_to is not None:
        env_ids = torch.nonzero(needs_latch, as_tuple=False).squeeze(-1)
        latch_command_state(env, env_ids=env_ids)
    if (env.moc_cmd_stamp < 0).all():
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    cur_xy = get_active_cube_pos_w(env)[:, :, :2]  # (N,3,2)
    ref_xy = env.moc_cmd_cube_pos_xy0              # (N,3,2)

    d = cur_xy - ref_xy
    dist = torch.linalg.vector_norm(d, dim=-1)        # (N,3)

    target = torch.clamp(env.target_cube_id, 0, 2)    # (N,)
    mask = torch.ones_like(dist, dtype=torch.bool)
    mask.scatter_(1, target.view(-1, 1), False)        # True for non-target cubes

    disturb = torch.clamp(dist - float(tol_xy), min=0.0)
    penalty = (disturb * mask.to(disturb.dtype)).sum(dim=1)  # (N,)

    return -float(lambda_disturb) * penalty


def reward_shaping_ee_to_target_xy(
    env: "ManagerBasedRLEnv",
    sigma_xy: float = 0.15,
    scale: float = 1.0,
) -> torch.Tensor:
    """Dense shaping: encourage EE to get close to target cube in XY (bounded in [0, scale])."""
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    # Need target id + cube poses
    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    # EE position (world)
    try:
        ee_frame = env.scene["ee_frame"]
    except KeyError:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)    
    
    ee_xy = ee_frame.data.target_pos_w[:, 0, :2]  # (N,2)


    cubes_xy = get_active_cube_pos_w(env)[:, :, :2]  # (N,3,2)
    tgt = torch.clamp(env.target_cube_id, 0, 2)      # (N,)

    N = env.num_envs
    tgt_xy = cubes_xy[torch.arange(N, device=env.device), tgt]  # (N,2)

    dist = torch.linalg.vector_norm(ee_xy - tgt_xy, dim=-1)  # (N,)
    # bounded shaping: 1 at dist=0, -> 0 as dist grows
    rew = torch.exp(-dist / float(sigma_xy))
    return float(scale) * rew


def reward_shaping_target_to_to_slot_xy(
    env: "ManagerBasedRLEnv",
    sigma_xy: float = 0.20,
    scale: float = 1.0,
) -> torch.Tensor:
    """Dense shaping: encourage target cube to approach the commanded to_slot in XY (bounded in [0, scale])."""
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    slots_xy = get_slots_w(env)[:, :, :2]  # (N,4,2)
    to_slot = torch.clamp(env.command_from_to[:, 1] - 1, 0, 3)  # (N,) 0..3

    N = env.num_envs
    to_xy = slots_xy[torch.arange(N, device=env.device), to_slot]  # (N,2)

    cubes_xy = get_active_cube_pos_w(env)[:, :, :2]  # (N,3,2)
    tgt = torch.clamp(env.target_cube_id, 0, 2)      # (N,)
    tgt_xy = cubes_xy[torch.arange(N, device=env.device), tgt]     # (N,2)

    dist = torch.linalg.vector_norm(tgt_xy - to_xy, dim=-1)  # (N,)
    rew = torch.exp(-dist / float(sigma_xy))
    return float(scale) * rew

def _get_suction_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Return (N,) float tensor in {0,1} indicating suction is active.
    Uses the same surface_gripper signal you already expose in observations.gripper_state().
    """
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        sg = env.scene.surface_grippers["surface_gripper"]
        # Typical state is in {-1,0,1}. Treat >0 as "suction on / engaged".
        return (sg.state.view(-1) > 0).to(torch.float32)
    # If no suction gripper exists, return zeros (reward disabled gracefully).
    return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)


# -------------------------
# Tool-success gating + lightweight logging helpers
# -------------------------

def _target_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,3) target cube world position."""
    cubes_pos = get_active_cube_pos_w(env)  # (N,3,3)
    tgt = torch.clamp(env.target_cube_id, 0, 2).to(torch.long)  # (N,)
    idx = torch.arange(env.num_envs, device=env.device)
    return cubes_pos[idx, tgt, :]

def _to_slot_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,3) to_slot world position."""
    slots = get_slots_w(env)  # (N,4,3)
    to_slot = torch.clamp(env.command_from_to[:, 1].to(torch.long) - 1, 0, 3)  # (N,)
    idx = torch.arange(env.num_envs, device=env.device)
    return slots[idx, to_slot, :]

def _is_lifted_target(env: "ManagerBasedRLEnv", z_lift_min: float = 0.03) -> torch.Tensor:
    """(N,) float {0,1} lifted indicator for target cube relative to table/slot height."""
    tgt_pos = _target_pos_w(env)  # (N,3)
    # Use slot Z as reference (fixed table height); slot_positions z ~ 0.021 in your cfg
    # If slots are per-env identical, this is stable.
    # Take current to_slot Z as reference height.
    to_pos = _to_slot_pos_w(env)  # (N,3)
    dz = tgt_pos[:, 2] - to_pos[:, 2]
    return (dz > float(z_lift_min)).to(torch.float32)

def _tool_success_mask(env: "ManagerBasedRLEnv", z_lift_min: float = 0.03) -> torch.Tensor:
    """(N,) bool: success achieved with suction active and lifted."""
    suction_on = _get_suction_state(env) > 0.0
    lifted = _is_lifted_target(env, z_lift_min=z_lift_min) > 0.0
    stable = _update_stable_success(env, stable_window=3)  # uses move_success internally
    return stable & suction_on & lifted

def _log_kv(env: "ManagerBasedRLEnv", key: str, value: torch.Tensor) -> None:
    """
    Store per-env tensors into env.extras so Sb3VecEnvWrapper can forward into info dict.
    Keys are flattened to 'moc/<name>'.
    Safe no-op if wrapper ignores extras; use --keep_all_info for guaranteed passthrough.
    """
    try:
        extras = getattr(env, "extras", None)
        if extras is None:
            extras = {}
            env.extras = extras
        # Detach for safety; keep tensor on device (wrapper may move/copy)
        extras[f"moc/{key}"] = value.detach()
    except Exception:
        # Never break training due to logging
        pass


def reward_shaping_ee_to_target_pregrasp_3d(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.20,
    scale: float = 1.0,
    z_offset: float = 0.08,
) -> torch.Tensor:
    """
    3D shaping: bring EE/suction tip to a pregrasp point above target cube.
    This creates a strong gradient that 'selects' the suction tip (not elbow).
    """
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    try:
        ee_frame = env.scene["ee_frame"]
    except KeyError:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    ee_pos = ee_frame.data.target_pos_w[:, 0, :3]  # (N,3)

    cubes_pos = get_active_cube_pos_w(env)  # (N,3,3)
    tgt = torch.clamp(env.target_cube_id, 0, 2)  # (N,)
    idx = torch.arange(env.num_envs, device=env.device)
    tgt_pos = cubes_pos[idx, tgt, :]  # (N,3)

    pregrasp = tgt_pos + torch.tensor([0.0, 0.0, float(z_offset)], device=env.device).view(1, 3)
    dist = torch.linalg.vector_norm(ee_pos - pregrasp, dim=-1)  # (N,)

    rew = torch.exp(-dist / float(sigma))
    return float(scale) * rew


def reward_suction_near_target(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.08,
    scale_proximity: float = 2.0,
    scale_bonus_if_suction_on: float = 2.0,
    body_name_regex_tip: str = r"(tool|ee|tcp|suction)",
) -> torch.Tensor:
    """
    Reward por acercar la PUNTA (tip) al cubo target.
    - Siempre da shaping (aunque suction esté OFF) => gradiente útil desde el inicio.
    - Da bonus adicional si suction está ON estando cerca (empuja a activar suction cerca).

    prox = exp(-dist/sigma)
    reward = scale_proximity*prox + scale_bonus_if_suction_on*suction_on*prox
    """
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    # Prefer ee_frame if exists; fallback to a tip link by regex
    tip_pos = None
    try:
        ee_frame = env.scene["ee_frame"]
        tip_pos = ee_frame.data.target_pos_w[:, 0, :3]
        tip_found = torch.ones((env.num_envs,), dtype=torch.float32, device=env.device)
    except KeyError:
        tip_pos, tip_found = _get_body_pos_by_regex(
            env,
            body_name_regex=body_name_regex_tip,
            cache_attr="_moc_tip_body_id",
        )

    if torch.all(tip_found == 0):
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    cubes_pos = get_active_cube_pos_w(env)  # (N,3,3)
    tgt = torch.clamp(env.target_cube_id, 0, 2).to(torch.long)
    idx = torch.arange(env.num_envs, device=env.device)
    tgt_pos = cubes_pos[idx, tgt, :]  # (N,3)

    dist = torch.linalg.vector_norm(tip_pos - tgt_pos, dim=-1)  # (N,)
    prox = torch.exp(-dist / float(sigma))

    suction_on = _get_suction_state(env)  # (N,) float {0,1}

    # Log
    _log_kv(env, "dist_tip_to_target_3d", dist)
    _log_kv(env, "prox_tip_to_target", prox)
    _log_kv(env, "suction_on", (suction_on > 0).to(torch.float32))

    return float(scale_proximity) * prox + float(scale_bonus_if_suction_on) * suction_on * prox


def reward_lift_target_when_suction(
    env: "ManagerBasedRLEnv",
    z_lift_min: float = 0.03,
    z_lift_max: float = 0.15,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Reward lifting the TARGET cube when suction is active.
    Height is measured relative to the slots' Z baseline (robust, no hardcoded table height).
    """
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    suction_on = _get_suction_state(env)  # (N,)

    cubes_pos = get_active_cube_pos_w(env)  # (N,3,3)
    slots = get_slots_w(env)               # (N,4,3)

    tgt = torch.clamp(env.target_cube_id, 0, 2)
    idx = torch.arange(env.num_envs, device=env.device)
    tgt_pos = cubes_pos[idx, tgt, :]       # (N,3)

    # Baseline Z from slots (approx table surface / slot centers)
    base_z = slots[:, :, 2].min(dim=1).values  # (N,)
    lift = tgt_pos[:, 2] - base_z             # (N,)

    # Normalize lift into [0,1] between z_lift_min and z_lift_max
    lift01 = (lift - float(z_lift_min)) / max(1e-6, (float(z_lift_max) - float(z_lift_min)))
    lift01 = torch.clamp(lift01, 0.0, 1.0)

    return float(scale) * suction_on * lift01


def reward_shaping_ee_to_target_pregrasp(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.20,
    scale: float = 1.0,
    z_offset: float = 0.08,
) -> torch.Tensor:
    """
    Shaping 3D: EE se acerca a un punto 'pregrasp' sobre el cubo objetivo.
    Esto suele ayudar a salir de políticas que no "entran" en contacto.
    """
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    try:
        ee_frame = env.scene["ee_frame"]
    except KeyError:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    ee_pos = ee_frame.data.target_pos_w[:, 0, :3]  # (N,3)

    cubes_pos = get_active_cube_pos_w(env)  # (N,3,3)
    tgt = torch.clamp(env.target_cube_id, 0, 2)  # (N,)
    N = env.num_envs
    tgt_pos = cubes_pos[torch.arange(N, device=env.device), tgt, :]  # (N,3)

    pregrasp = tgt_pos + torch.tensor([0.0, 0.0, float(z_offset)], device=env.device).view(1, 3)

    dist = torch.linalg.vector_norm(ee_pos - pregrasp, dim=-1)  # (N,)
    rew = torch.exp(-dist / float(sigma))
    return float(scale) * rew


def reward_shaping_target_to_to_slot_xy_gated_by_suction(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.20,
    scale: float = 1.0,
    z_lift_min: float = 0.03,
) -> torch.Tensor:
    """
    Shaping target->to_slot (XY), pero SOLO paga si:
      - suction está activo
      - y el target está levantado (evita arrastre/empush con suction "encendido")
    """
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    if not hasattr(env, "command_from_to") or env.command_from_to is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    suction_on = _get_suction_state(env)  # (N,) float {0,1}
    lifted = _is_lifted_target(env, z_lift_min=z_lift_min)  # (N,) float {0,1}
    gate = suction_on * lifted  # (N,)

    # Early-out if nobody is gated-in
    if torch.all(gate == 0):
        return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    tgt_pos = _target_pos_w(env)        # (N,3)
    to_pos = _to_slot_pos_w(env)        # (N,3)

    dist_xy = torch.linalg.vector_norm(tgt_pos[:, :2] - to_pos[:, :2], dim=-1)  # (N,)
    rew = torch.exp(-dist_xy / float(sigma))  # (N,)

    # Optional: log a couple of diagnostics (per-env)
    _log_kv(env, "dist_target_to_to_xy", dist_xy)
    _log_kv(env, "gate_transport", gate)

    return float(scale) * gate * rew


def reward_log_metrics(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Weight=0. Solo side-effects: registra métricas por-env para TensorBoard.
    """
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    zeros = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    # Basic signals
    suction = (_get_suction_state(env) > 0).to(torch.float32)
    lifted = _is_lifted_target(env, z_lift_min=0.03)
    stable = _update_stable_success(env, stable_window=3).to(torch.float32)
    tool_succ = (stable > 0) * suction * (lifted > 0).to(torch.float32)

    # Distances
    try:
        ee_frame = env.scene["ee_frame"]
        ee_pos = ee_frame.data.target_pos_w[:, 0, :3]
        tgt_pos = _target_pos_w(env)
        to_pos = _to_slot_pos_w(env)
        dist_ee_tgt = torch.linalg.vector_norm(ee_pos - tgt_pos, dim=-1)
        dist_tgt_to_xy = torch.linalg.vector_norm(tgt_pos[:, :2] - to_pos[:, :2], dim=-1)
    except Exception:
        dist_ee_tgt = zeros
        dist_tgt_to_xy = zeros

    _log_kv(env, "suction_on", suction)
    _log_kv(env, "lifted", lifted)
    _log_kv(env, "stable_success", stable)
    _log_kv(env, "tool_success", tool_succ)
    _log_kv(env, "dist_ee_to_target_3d", dist_ee_tgt)
    _log_kv(env, "dist_target_to_to_xy", dist_tgt_to_xy)

    return zeros

def _get_body_pos_by_regex(
    env: "ManagerBasedRLEnv",
    body_name_regex: str,
    cache_attr: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      pos_w: (N,3) body world position (zeros if not found)
      found: (N,) float {0,1}
    Uses robot.data.body_pos_w and caches body index on env.
    """
    robot = env.scene["robot"]
    zeros = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)

    body_id = getattr(env, cache_attr, None)
    if body_id is None:
        # body_names is a list[str]
        names = list(robot.data.body_names)
        pat = re.compile(body_name_regex)
        matches = [i for i, n in enumerate(names) if pat.search(n)]
        if len(matches) == 0:
            setattr(env, cache_attr, -1)
            return zeros, torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
        setattr(env, cache_attr, int(matches[0]))
        body_id = int(matches[0])

    if int(body_id) < 0:
        return zeros, torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    pos = robot.data.body_pos_w[:, int(body_id), :3]
    return pos, torch.ones((env.num_envs,), dtype=torch.float32, device=env.device)


def reward_penalize_elbow_near_cubes(
    env: "ManagerBasedRLEnv",
    safe_in_slot_pitches: float = 0.35,
    penalty_scale: float = 10.0,
    body_name_regex: str = r"(elbow|forearm)",
) -> torch.Tensor:
    """
    Penaliza si el codo se acerca a menos de safety_dist, definida como:
      safety_dist = safe_in_slot_pitches * (min distancia XY entre slots)

    Esto evita hardcodear metros y se adapta a tu escena.
    """
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    zeros = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    elbow_pos, found = _get_body_pos_by_regex(env, body_name_regex, "_moc_elbow_body_id")
    if torch.all(found == 0):
        return zeros

    # Compute slot pitch (XY) per env from slots
    slots = get_slots_w(env)  # (N,4,3)
    sxy = slots[:, :, :2]     # (N,4,2)
    # pairwise distances (4x4); take min of upper triangle
    d01 = torch.linalg.vector_norm(sxy[:, 0] - sxy[:, 1], dim=-1)
    d02 = torch.linalg.vector_norm(sxy[:, 0] - sxy[:, 2], dim=-1)
    d03 = torch.linalg.vector_norm(sxy[:, 0] - sxy[:, 3], dim=-1)
    d12 = torch.linalg.vector_norm(sxy[:, 1] - sxy[:, 2], dim=-1)
    d13 = torch.linalg.vector_norm(sxy[:, 1] - sxy[:, 3], dim=-1)
    d23 = torch.linalg.vector_norm(sxy[:, 2] - sxy[:, 3], dim=-1)
    pitch = torch.minimum(torch.minimum(torch.minimum(d01, d02), torch.minimum(d03, d12)), torch.minimum(d13, d23))

    safety_dist = float(safe_in_slot_pitches) * pitch  # (N,)

    cubes_pos = get_active_cube_pos_w(env)  # (N,3,3)
    d = torch.linalg.vector_norm(cubes_pos - elbow_pos.unsqueeze(1), dim=-1)  # (N,3)
    dmin = d.min(dim=1).values

    margin = torch.clamp(safety_dist - dmin, min=0.0)
    rel = margin / torch.clamp(safety_dist, min=1e-6)
    penalty = -float(penalty_scale) * (rel * rel)

    _log_kv(env, "elbow_safety_dist", safety_dist)
    _log_kv(env, "elbow_dmin_to_cubes", dmin)
    _log_kv(env, "elbow_penalty", penalty)

    return penalty