from __future__ import annotations

import torch
from typing import Dict, Any, Optional

from isaaclab.envs import ManagerBasedRLEnv

from .terminations import move_success
from .rewards import reward_penalty_disturb_other_cubes


@torch.no_grad()
def collect_step_metrics(
    env: ManagerBasedRLEnv,
    *,
    tol_xy: float = 0.02,
    tol_z: float = 0.05,
    require_to_clear: bool = True,
    require_settled: bool = False,
    vel_tol: float = 0.20,
    disturb_tol_xy: float = 0.01,
) -> Dict[str, Any]:
    """
    Compute per-env metrics tensors for debugging/training monitoring.
    Returns tensors on env.device.
    """
    ok = move_success(
        env,
        tol_xy=tol_xy,
        tol_z=tol_z,
        require_to_clear=require_to_clear,
        require_settled=require_settled,
        vel_tol=vel_tol,
    )

    disturb = reward_penalty_disturb_other_cubes(
        env,
        lambda_disturb=1.0,  # return raw magnitude (you scale outside)
        tol_xy=disturb_tol_xy,
    )

    next_signal = getattr(env, "moc_next_signal", None)
    if next_signal is None:
        next_signal = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)

    cmd = getattr(env, "command_from_to", None)
    if cmd is None:
        cmd = torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)

    tgt = getattr(env, "target_cube_id", None)
    if tgt is None:
        tgt = -torch.ones((env.num_envs,), dtype=torch.long, device=env.device)

    return {
        "move_success": ok.to(torch.bool),
        "disturb_raw": disturb.to(torch.float32),
        "next_signal": next_signal.to(torch.float32),
        "cmd_from_to": cmd.to(torch.long),
        "target_cube_id": tgt.to(torch.long),
    }


@torch.no_grad()
def summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert per-env tensors to scalar summary dict (python floats).
    """
    out: Dict[str, float] = {}

    ok = metrics["move_success"].float()
    out["move_success_rate"] = ok.mean().item()

    disturb = metrics["disturb_raw"]
    out["disturb_mean"] = disturb.mean().item()
    out["disturb_p95"] = torch.quantile(disturb, 0.95).item()

    ns = metrics["next_signal"]
    out["next_signal_mean"] = ns.mean().item()
    out["next_signal_rate_pos"] = (ns > 0.0).float().mean().item()

    return out