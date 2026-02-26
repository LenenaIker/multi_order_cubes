from __future__ import annotations

import time
import numpy as np
import torch
from typing import Dict, Any
import inspect

from isaaclab.envs import ManagerBasedRLEnv

from multi_order_cubes.mdp.diagnostics import collect_step_metrics, summarize_metrics
from multi_order_cubes.mdp.step_cache import (
    get_nearest_slot_for_active_cubes_xy,
    get_active_cube_pos_w,
    get_slots_w
)
from multi_order_cubes.mdp.terminations import move_success
from multi_order_cubes.mdp.commands import (
    ensure_command_buffer,
    ensure_moc_buffers,
    latch_command_state
    
)

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

@torch.no_grad()
def run_no_false_positive_adjacent_slots_smoke(
    env: ManagerBasedRLEnv,
    trials: int = 5,
    tol_xy: float = 0.02,
    tol_z: float = 0.05,
    clear_tol_xy: float = 0.08,
) -> None:
    """
    Detecta el bug clásico: si el success usa mal tol_xy (p.ej. acepta ~0.2m),
    entonces al poner el comando (from=vecino, to=slot vacío) el target está a 0.2m
    y move_success podría dar True en reset. Con el fix, debe ser siempre False.

    En tu layout, los slots están a y = {0.3, 0.1, -0.1, -0.3} (delta 0.2m).
    """
    print("\n[NO-FALSE-POSITIVE-ADJACENT] start")
    ensure_command_buffer(env)
    ensure_moc_buffers(env)

    bad_total = 0

    for k in range(trials):
        obs, info = env.reset()

        # Occupancy por nearest-slot (activos)
        nearest = get_nearest_slot_for_active_cubes_xy(env)  # (N,3)
        occ = torch.zeros((env.num_envs, 4), dtype=torch.bool, device=env.device)
        occ.scatter_(1, nearest, True)                        # (N,4)
        empty_mask = ~occ                                     # (N,4)

        # Como hay 3 cubos + 4 slots, debe haber 1 vacío por env (en condiciones normales).
        empty_idx = empty_mask.to(torch.int64).argmax(dim=1)  # (N,) in [0..3]

        # Elegimos FROM como vecino del slot vacío:
        # - si empty=0 -> from=1
        # - si empty=3 -> from=2
        # - si empty=1/2 -> from=empty-1
        from_idx = torch.where(
            empty_idx == 0,
            torch.ones_like(empty_idx),
            torch.where(empty_idx == 3, 2 * torch.ones_like(empty_idx), empty_idx - 1),
        )  # (N,) in [0..3], garantizando vecino

        # Set command per-env (1-based)
        env.command_from_to[:, 0] = from_idx + 1
        env.command_from_to[:, 1] = empty_idx + 1

        # Latch target/baselines acorde al comando
        latch_command_state(env)

        # Evaluar success inmediatamente (sin mover robot)
        ms = move_success(
            env,
            tol_xy=tol_xy,
            tol_z=tol_z,
            require_to_clear=True,
            clear_tol_xy=clear_tol_xy,
            require_settled=False,
        )

        bad = int(ms.sum().item())
        bad_total += bad

        print(f"  trial={k+1}/{trials}  move_success_true={bad}/{env.num_envs}")

        if bad > 0:
            # Debug mínimo: imprime algunos env_ids fallidos
            bad_ids = torch.nonzero(ms).squeeze(-1)[:10]
            print(f"    BAD env_ids (first 10): {bad_ids.tolist()}")

    if bad_total > 0:
        raise AssertionError(
            f"[NO-FALSE-POSITIVE-ADJACENT] FAIL: move_success was True {bad_total} times "
            f"across {trials} trials. Esto sugiere falso positivo (tol_xy)."
        )

    print("[NO-FALSE-POSITIVE-ADJACENT] PASS")


@torch.no_grad()
def run_require_settled_no_crash_smoke(
    env,
    trials: int = 3,
) -> None:
    """
    Verifica que move_success(require_settled=True) no crashea.
    No depende de los nombres exactos de los parámetros de settled.
    """
    print("\n[REQUIRE-SETTLED-NO-CRASH] start")

    sig = inspect.signature(move_success)
    param_names = set(sig.parameters.keys())

    # kwargs base (deben existir en tu move_success sí o sí)
    base_kwargs = dict(
        tol_xy=0.02,
        tol_z=0.05,
        require_to_clear=True,
        clear_tol_xy=0.08,
        require_settled=True,
    )

    # kwargs opcionales: intenta varios nombres comunes
    settled_candidates = [
        ("settled_lin_vel", 0.05),
        ("settled_ang_vel", 0.6),
        ("settled_lin_vel_w", 0.05),
        ("settled_ang_vel_w", 0.6),
        ("lin_vel_thresh", 0.05),
        ("ang_vel_thresh", 0.6),
        ("settled_lin_vel_thresh", 0.05),
        ("settled_ang_vel_thresh", 0.6),
        ("lin_vel_threshold", 0.05),
        ("ang_vel_threshold", 0.6),
    ]

    # Filtra: solo pasa los kwargs que existan en la firma real
    extra_kwargs = {}
    for k, v in settled_candidates:
        if k in param_names:
            extra_kwargs[k] = v

    # Nota: si tu move_success usa nombres distintos, igualmente no crashea;
    # solo no ajusta thresholds y usa defaults internos.
    if extra_kwargs:
        print(f"  Using settled kwargs: {extra_kwargs}")
    else:
        print("  No settled-threshold kwargs found in move_success signature; using defaults.")

    for k in range(trials):
        env.reset()
        ms = move_success(env, **base_kwargs, **extra_kwargs)

        assert ms.shape == (env.num_envs,), f"Unexpected move_success shape: {ms.shape}"
        assert ms.dtype == torch.bool, f"Unexpected move_success dtype: {ms.dtype}"
        print(f"  trial={k+1}/{trials} ok (move_success_true={int(ms.sum().item())}/{env.num_envs})")

    print("[REQUIRE-SETTLED-NO-CRASH] PASS")


@torch.no_grad()
def run_step_cache_invalidation_without_next_smoke(
    env: ManagerBasedRLEnv,
    steps: int = 20,
) -> None:
    """
    Verifica que el cache se invalida entre steps incluso si no activas NEXT.
    Depende de que tu fix haya introducido un token basado en episode_length_buf (o similar).
    """
    print("\n[STEP-CACHE-INVALIDATION-WITHOUT-NEXT] start")

    env.reset()

    act_dim = int(env.action_manager.total_action_dim)
    next_index = act_dim - 1

    # Fuerza cache population
    _ = get_active_cube_pos_w(env)

    if not hasattr(env, "_moc_cache_token"):
        raise AssertionError(
            "env._moc_cache_token no existe. El fix del step_cache no parece aplicado."
        )

    tok_prev = int(env._moc_cache_token)
    changed = 0

    for t in range(steps):
        actions = torch.zeros((env.num_envs, act_dim), device=env.device)
        actions[:, next_index] = -1.0  # “no NEXT”

        env.step(actions)

        # Re-populate + force invalidation check
        _ = get_active_cube_pos_w(env)
        tok_now = int(env._moc_cache_token)

        if tok_now != tok_prev:
            changed += 1
        tok_prev = tok_now

    if changed == 0:
        raise AssertionError(
            "[STEP-CACHE-INVALIDATION-WITHOUT-NEXT] FAIL: cache token never changed across steps. "
            "Esto sugiere que el cache podría quedarse stale."
        )

    print(f"[STEP-CACHE-INVALIDATION-WITHOUT-NEXT] PASS (token changed {changed}/{steps} steps)")

@torch.no_grad()
def run_move_success_reset_diagnostics(
    env: ManagerBasedRLEnv,
    trials: int = 5,
    max_print_envs: int = 6,
    tol_xy: float = 0.02,
    tol_z: float = 0.05,
    clear_tol_xy: float = 0.08,
    require_settled: bool = True,
) -> None:
    """
    Si move_success devuelve True en reset, imprime métricas para decidir si es TP o FP.
    """
    print("\n[MOVE-SUCCESS-RESET-DIAGNOSTICS] start")

    for k in range(trials):
        env.reset()

        ms = move_success(
            env,
            tol_xy=tol_xy,
            tol_z=tol_z,
            require_to_clear=True,
            clear_tol_xy=clear_tol_xy,
            require_settled=require_settled,
        )

        n_true = int(ms.sum().item())
        print(f"  trial={k+1}/{trials} move_success_true={n_true}/{env.num_envs}")

        if n_true == 0:
            continue

        # env ids con success
        env_ids = torch.nonzero(ms).squeeze(-1)[:max_print_envs]
        print(f"    env_ids: {env_ids.tolist()}")

        # Necesitamos: posiciones de cubos activos (N,3,3) y slots (N,4,3)
        cubes_w = get_active_cube_pos_w(env)   # (N,3,3)
        slots_w = get_slots_w(env)             # (N,4,3)

        # to_slot_idx (0-based) desde command buffer (1-based)
        to_idx = (env.command_from_to[:, 1].to(torch.long) - 1).clamp(0, 3)  # (N,)

        # target cube: si existe target_cube_id (0..2), úsalo, si no fallback: nearest to from_slot
        if hasattr(env, "target_cube_id") and env.target_cube_id is not None:
            tgt = env.target_cube_id.to(torch.long).clamp(0, 2)  # (N,)
        else:
            # fallback: nearest cube to from_slot
            from_idx = (env.command_from_to[:, 0].to(torch.long) - 1).clamp(0, 3)
            from_pos = slots_w[torch.arange(env.num_envs, device=env.device), from_idx]  # (N,3)
            dxy = cubes_w[:, :, :2] - from_pos[:, None, :2]                               # (N,3,2)
            tgt = (dxy[..., 0] ** 2 + dxy[..., 1] ** 2).argmin(dim=1)                     # (N,)

        # Extraer target pos y to_slot pos
        tgt_pos = cubes_w[torch.arange(env.num_envs, device=env.device), tgt]             # (N,3)
        to_pos  = slots_w[torch.arange(env.num_envs, device=env.device), to_idx]          # (N,3)

        dx = tgt_pos[:, 0] - to_pos[:, 0]
        dy = tgt_pos[:, 1] - to_pos[:, 1]
        dz = tgt_pos[:, 2] - to_pos[:, 2]

        dist_xy = torch.sqrt(dx * dx + dy * dy)  # (N,)
        abs_dz  = dz.abs()

        # “to_slot libre”: min dist_xy de otros cubos al to_slot
        all_dxy = cubes_w[:, :, :2] - to_pos[:, None, :2]              # (N,3,2)
        all_d2  = all_dxy[..., 0] ** 2 + all_dxy[..., 1] ** 2          # (N,3)
        mask_other = torch.ones_like(all_d2, dtype=torch.bool)
        mask_other[torch.arange(env.num_envs, device=env.device), tgt] = False
        other_min_xy = torch.sqrt(all_d2.masked_fill(~mask_other, float("inf")).min(dim=1).values)

        for eid in env_ids.tolist():
            print(
                f"    env={eid}  tgt={int(tgt[eid])}  to_slot={int(to_idx[eid])}  "
                f"dist_xy={float(dist_xy[eid]):.4f}  abs_dz={float(abs_dz[eid]):.4f}  "
                f"other_min_xy={float(other_min_xy[eid]):.4f}"
            )

    print("[MOVE-SUCCESS-RESET-DIAGNOSTICS] done")

@torch.no_grad()
def run_no_start_in_success_smoke(env: ManagerBasedRLEnv, trials: int = 20) -> None:
    print("\n[NO-START-IN-SUCCESS] start")
    total = 0
    for k in range(trials):
        env.reset()
        ms = move_success(
            env,
            tol_xy=0.02,
            tol_z=0.05,
            require_to_clear=True,
            clear_tol_xy=0.08,
            require_settled=False,
        )
        n = int(ms.sum().item())
        total += n
        print(f"  trial={k+1}/{trials} move_success_true={n}/{env.num_envs}")
    if total > 0:
        raise AssertionError(f"[NO-START-IN-SUCCESS] FAIL: saw {total} successes at reset across {trials} trials.")
    print("[NO-START-IN-SUCCESS] PASS")