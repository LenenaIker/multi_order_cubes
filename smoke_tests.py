from __future__ import annotations

import time
import numpy as np
import torch
from typing import Dict, Any

from isaaclab.envs import ManagerBasedRLEnv

from multi_order_cubes.mdp.diagnostics import collect_step_metrics, summarize_metrics


def _print_summary(tag: str, summary: Dict[str, float], steps: int, dt: float) -> None:
    sps = steps / max(dt, 1e-6)
    print(f"\n[{tag}] steps={steps}  wall={dt:.2f}s  steps/s={sps:.1f}")
    for k, v in summary.items():
        print(f"  - {k}: {v:.6f}")


@torch.no_grad()
def run_random_policy_smoke(
    env: ManagerBasedRLEnv,
    steps: int = 4000,
    seed: int = 0,
    metrics_every: int = 500,
) -> None:
    """
    Random actions to check:
      - move_success almost never true
      - disturb not always 0, not exploding
      - NEXT not trivially giving success
    """
    obs, info = env.reset()
    np.random.seed(seed)

    t0 = time.time()
    acc = None
    n_acc = 0

    for t in range(steps):
        # Sample actions matching action space shape.
        # IsaacLab uses torch actions on env.device.
        act_dim = int(env.action_manager.total_action_dim)
        actions = torch.rand((env.num_envs, act_dim), device=env.device) * 2.0 - 1.0

        obs, rew, terminated, truncated, info = env.step(actions)

        m = collect_step_metrics(env)
        # accumulate summaries cheaply
        sm = summarize_metrics(m)
        if acc is None:
            acc = {k: 0.0 for k in sm}
        for k, v in sm.items():
            acc[k] += float(v)
        n_acc += 1

        if (t + 1) % metrics_every == 0:
            avg = {k: acc[k] / max(n_acc, 1) for k in acc}
            _print_summary("RANDOM", avg, t + 1, time.time() - t0)
            acc, n_acc = None, 0

        # optional: auto-reset if your env needs it
        if torch.any(terminated | truncated):
            env.reset()

    # Final
    if acc is not None:
        avg = {k: acc[k] / max(n_acc, 1) for k in acc}
        _print_summary("RANDOM-FINAL", avg, steps, time.time() - t0)


@torch.no_grad()
def run_adversarial_next_spam_smoke(
    env: ManagerBasedRLEnv,
    steps: int = 2000,
    metrics_every: int = 500,
) -> None:
    """
    "Adversarial" test: spam NEXT (if NEXT is part of action vector),
    expecting:
      - move_success stays low
      - next_signal_rate_pos ~1 (if we set it)
      - disturb not necessarily high
    """
    obs, info = env.reset()
    t0 = time.time()

    acc = None
    n_acc = 0

    # Build actions: zeros except NEXT dim set high.
    # Assumption: NEXT is last dim. If not, you’ll adjust index.
    act_dim = int(env.action_manager.total_action_dim)
    
    next_index = act_dim - 1  # adjust if NEXT not last

    for t in range(steps):
        actions = torch.zeros((env.num_envs, act_dim), device=env.device)
        actions[:, next_index] = 1.0  # spam NEXT

        obs, rew, terminated, truncated, info = env.step(actions)

        m = collect_step_metrics(env)
        sm = summarize_metrics(m)
        if acc is None:
            acc = {k: 0.0 for k in sm}
        for k, v in sm.items():
            acc[k] += float(v)
        n_acc += 1

        if (t + 1) % metrics_every == 0:
            avg = {k: acc[k] / max(n_acc, 1) for k in acc}
            _print_summary("NEXT-SPAM", avg, t + 1, time.time() - t0)
            acc, n_acc = None, 0

        if torch.any(terminated | truncated):
            env.reset()

    if acc is not None:
        avg = {k: acc[k] / max(n_acc, 1) for k in acc}
        _print_summary("NEXT-SPAM-FINAL", avg, steps, time.time() - t0)


@torch.no_grad()
def run_disturb_push_smoke(
    env: ManagerBasedRLEnv,
    steps: int = 2000,
    metrics_every: int = 500,
) -> None:
    """
    Try to induce lateral contacts by aggressively moving actions (still random-ish),
    expecting disturb_mean > 0 at least sometimes.
    """
    obs, info = env.reset()
    t0 = time.time()
    acc = None
    n_acc = 0

    if not hasattr(env.action_space, "shape") or env.action_space.shape is None:
        raise RuntimeError("Need env.action_space.shape for disturb push test.")

    act_dim = int(env.action_manager.total_action_dim)
    # Assumption: NEXT is last dim; keep it 0 to avoid advancing commands.
    next_index = act_dim - 1

    for t in range(steps):
        actions = torch.rand((env.num_envs, act_dim), device=env.device) * 2.0 - 1.0
        actions[:, next_index] = -1.0  # disable NEXT

        # amplify first few dims to make robot swing harder (if those are Cartesian deltas)
        # If your action mapping is different, this still tests "do we ever disturb?"
        actions[:, : min(6, act_dim - 1)] *= 1.0

        obs, rew, terminated, truncated, info = env.step(actions)

        m = collect_step_metrics(env)
        sm = summarize_metrics(m)
        if acc is None:
            acc = {k: 0.0 for k in sm}
        for k, v in sm.items():
            acc[k] += float(v)
        n_acc += 1

        if (t + 1) % metrics_every == 0:
            avg = {k: acc[k] / max(n_acc, 1) for k in acc}
            _print_summary("DISTURB-PUSH", avg, t + 1, time.time() - t0)
            acc, n_acc = None, 0

        if torch.any(terminated | truncated):
            env.reset()

    if acc is not None:
        avg = {k: acc[k] / max(n_acc, 1) for k in acc}
        _print_summary("DISTURB-PUSH-FINAL", avg, steps, time.time() - t0)
