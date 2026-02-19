# mdp/rewards.py (APPEND THIS BLOCK)

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .terminations import move_success
from .commands import sample_command_from_to, ensure_command_buffer


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

    commit_ok = next_evt & stable  # (N,)
    rew = commit_ok.to(torch.float32) * float(R_commit)

    # Apply cooldown on any NEXT attempt (ok or fail)
    _trigger_cooldown(env, next_evt, cooldown_steps)

    # Advance ONLY those envs that committed correctly
    if advance_command and commit_ok.any():
        env_ids = torch.nonzero(commit_ok, as_tuple=False).squeeze(-1)
        sample_command_from_to(env, env_ids=env_ids)  # <-- requires commands.py patch below

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

