"""Pure-Python, Isaac-Sim-free diagnostics for play_sb3.py.

No isaaclab / stable_baselines3 imports here on purpose: everything below takes plain
tensors (already .detach()'d) or numpy/Python scalars and returns numpy/Python data, so
this module can be imported and exercised with fabricated tensors in a bare `python`
process, without launching AppLauncher, as a sanity check before ever running the real
rollout.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SIZE_NAMES = ("s", "m", "l")             # idx % 3 into CUBE_KEYS_9
COLOR_NAMES = ("light", "flat", "dark")  # idx // 3 into CUBE_KEYS_9
STATE_NAMES = ("not_trying", "hovering", "attempting", "grasping")

# Match reward_grasp_contact / reward_next_signal defaults (mdp/rewards.py) for consistency.
SUCCESS_XY = 0.05
SUCCESS_Z = 0.03
# Match diag_grip_distance's own closed_threshold (mdp/rewards.py) for consistency.
CLOSED_THRESHOLD = 0.7


@dataclass(frozen=True)
class CubeStateSnapshot:
    """Cloned copy of env cube-identity attributes, taken BEFORE env.step(action).

    Must be built from .detach().clone()'d tensors: `active_cube_indices`,
    `moc_active_cube_slot_idx`, `moc_slot_to_active_id`, and `target_cube_id` are all
    mutated in place inside env.step() (on reset AND on a mid-episode NEXT press -- see
    mdp/events.py::randomize_cubes_on_slots and mdp/commands.py::latch_target_cube_from_command),
    so a bare reference here would silently observe post-step values instead of the
    pre-step ones the about-to-be-computed reward actually corresponds to.
    """

    active_cube_indices: torch.Tensor    # (num_envs, 3) long, 0-8 index into CUBE_KEYS_9
    active_cube_slot_idx: torch.Tensor   # (num_envs, 3) long, which of 4 slots each active id sits in
    slot_to_active_id: torch.Tensor      # (num_envs, 4) long, -1 or active id
    target_cube_id: torch.Tensor         # (num_envs,) long, 0-2


def snapshot_cube_state(env) -> CubeStateSnapshot:
    """Call on the raw (unwrapped) ManagerBasedRLEnv, BEFORE env.step(action)."""
    return CubeStateSnapshot(
        active_cube_indices=env.active_cube_indices.detach().clone(),
        active_cube_slot_idx=env.moc_active_cube_slot_idx.detach().clone(),
        slot_to_active_id=env.moc_slot_to_active_id.detach().clone(),
        target_cube_id=env.target_cube_id.detach().clone(),
    )


def derive_target_facts(snap: CubeStateSnapshot) -> dict[str, np.ndarray]:
    """Vectorized per-env derivation of target size/color/slot/neighbor occupancy.

    Returns numpy arrays, each shape (num_envs,):
      size_id (0/1/2), color_id (0/1/2), slot_id (0-3),
      left_occupied (0/1, or -1 meaning "no left neighbor, edge slot"),
      right_occupied (0/1, or -1 meaning "no right neighbor, edge slot").
    """
    row = torch.arange(snap.target_cube_id.shape[0], device=snap.target_cube_id.device)
    target_cube_idx = snap.active_cube_indices[row, snap.target_cube_id]  # (num_envs,), 0-8
    size_id = (target_cube_idx % 3).cpu().numpy()
    color_id = (target_cube_idx // 3).cpu().numpy()
    slot_id = snap.active_cube_slot_idx[row, snap.target_cube_id].cpu().numpy()

    slot_to_active = snap.slot_to_active_id.cpu().numpy()  # (num_envs, 4)
    n = slot_to_active.shape[0]
    left_occupied = np.full(n, -1, dtype=np.int8)
    right_occupied = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        s = int(slot_id[i])
        if s > 0:
            left_occupied[i] = 1 if slot_to_active[i, s - 1] >= 0 else 0
        if s < 3:
            right_occupied[i] = 1 if slot_to_active[i, s + 1] >= 0 else 0

    return dict(
        size_id=size_id, color_id=color_id, slot_id=slot_id,
        left_occupied=left_occupied, right_occupied=right_occupied,
    )


def classify_state(
    dist_xy: np.ndarray, dist_z: np.ndarray, closed_frac: np.ndarray, grasp_contact_reward: np.ndarray
) -> np.ndarray:
    """Three-plus-one-way behavioral classification, vectorized over equal-shape arrays.

    Returns an int8 array: 0 = not_trying, 1 = hovering, 2 = attempting, 3 = grasping.
    "not_trying" (outside the reach success zone entirely) is reported separately from
    "hovering" (in position but not closing) to distinguish "never tries" from "tries and
    fails" -- a materially different diagnosis for the hover-fear hypothesis.
    """
    in_position = (dist_xy < SUCCESS_XY) & (dist_z < SUCCESS_Z)
    closing = closed_frac > CLOSED_THRESHOLD
    grasping = grasp_contact_reward > 0.0

    state = np.zeros_like(dist_xy, dtype=np.int8)          # not_trying
    state[in_position & ~closing] = 1                       # hovering
    state[in_position & closing & ~grasping] = 2             # attempting
    state[in_position & closing & grasping] = 3              # grasping
    return state


_ASSIGNMENT_KEYS = ("slot_id", "size_id", "color_id", "left_occupied", "right_occupied")


def build_rows(
    step_idx: int, snap: CubeStateSnapshot, extras: dict, sb3_rewards, dones,
    prev_facts: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """One dict per env for this step, ready to append to the in-memory row buffer.

    Also returns this call's `facts` dict so the caller can pass it back in as `prev_facts`
    on the NEXT call: comparing consecutive facts is how `is_new_assignment` detects a fresh
    command latch (mid-episode NEXT resample or a reset), independent of dwell time -- see
    `aggregate_and_print`'s Table 4, which uses this to separate "how often does this bucket
    get assigned" from "how many steps pile up in it" (row count conflates the two: a bucket
    the policy can't escape accumulates rows for the rest of the episode even if it's rarely
    assigned in the first place).
    """
    facts = derive_target_facts(snap)
    n = snap.target_cube_id.shape[0]

    if prev_facts is None:
        is_new_assignment = np.ones(n, dtype=bool)
    else:
        is_new_assignment = np.zeros(n, dtype=bool)
        for key in _ASSIGNMENT_KEYS:
            is_new_assignment |= facts[key] != prev_facts[key]

    def e(key: str, default: float = 0.0) -> np.ndarray:
        t = extras.get(key)
        return t.detach().cpu().numpy() if t is not None else np.full(n, default, dtype=np.float32)

    dist_xy = e("position/dist_xy")
    dist_z = e("position/dist_z")
    closed_frac = e("grip/closed_frac")
    pinch_cos_sim = e("grip/pinch_cos_sim")
    reach_xy = e("rewards/reach_xy")
    reach_z = e("rewards/reach_z")
    grip_readiness = e("rewards/grip_readiness")
    grasp_contact = e("rewards/grasp_contact")
    object_lifted = e("rewards/object_lifted")
    state = classify_state(dist_xy, dist_z, closed_frac, grasp_contact)

    rows = []
    for i in range(n):
        rows.append(dict(
            step=step_idx, env=i, done=bool(dones[i]),
            size_id=int(facts["size_id"][i]), color_id=int(facts["color_id"][i]),
            slot_id=int(facts["slot_id"][i]),
            left_occupied=int(facts["left_occupied"][i]), right_occupied=int(facts["right_occupied"][i]),
            is_new_assignment=bool(is_new_assignment[i]),
            state=int(state[i]),
            dist_xy=float(dist_xy[i]), dist_z=float(dist_z[i]),
            closed_frac=float(closed_frac[i]), pinch_cos_sim=float(pinch_cos_sim[i]),
            reach_xy=float(reach_xy[i]), reach_z=float(reach_z[i]),
            grip_readiness=float(grip_readiness[i]),
            grasp_contact=float(grasp_contact[i]),
            object_lifted=float(object_lifted[i]),
            total_reward=float(sb3_rewards[i]),
        ))
    return rows, facts


_REWARD_COLS = ("reach_xy", "reach_z", "grip_readiness", "grasp_contact", "object_lifted", "total_reward")


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total > 0 else "   --"


def _mean(rows: list[dict], col: str) -> str:
    return f"{sum(r[col] for r in rows) / len(rows):8.3f}" if rows else "      --"


def _print_table(title: str, header: str, group_rows: dict, total_n: int, extra_cols=()) -> None:
    print(f"\n=== {title} ===")
    print(header)
    for key in sorted(group_rows.keys(), key=str):
        rows = group_rows[key]
        n = len(rows)
        line = f"{str(key):<12} {n:>7} {_pct(n, total_n)} "
        for col in extra_cols:
            line += f"{_mean(rows, col)} "
        print(line)


def aggregate_and_print(rows: list[dict]) -> None:
    """Pure stdlib groupby (no pandas) producing the four requested breakdowns."""
    if not rows:
        print("No diagnostic rows collected, nothing to summarize.")
        return

    total_n = len(rows)

    # --- Table 1: reward by behavioral state ---
    by_state: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_state[r["state"]].append(r)
    print("\n=== Reward by behavioral state (does hovering already pay off?) ===")
    header = f"{'state':<12} {'n_rows':>7} {'pct':>6} " + " ".join(f"{c:>8}" for c in _REWARD_COLS)
    print(header)
    for s in range(4):
        rs = by_state.get(s, [])
        n = len(rs)
        line = f"{STATE_NAMES[s]:<12} {n:>7} {_pct(n, total_n)} "
        line += " ".join(_mean(rs, c) for c in _REWARD_COLS)
        print(line)

    # --- Table 2: outcome rate by target cube size ---
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_size[r["size_id"]].append(r)
    print("\n=== Outcome rate by target cube size ===")
    print(f"{'size':<12} {'n_rows':>7} " + " ".join(f"{s:>10}" for s in STATE_NAMES) + f" {'mean_total':>10}")
    for sid in range(3):
        rs = by_size.get(sid, [])
        n = len(rs)
        state_counts = [sum(1 for r in rs if r["state"] == s) for s in range(4)]
        line = f"{SIZE_NAMES[sid]:<12} {n:>7} "
        line += " ".join(f"{_pct(c, n):>10}" for c in state_counts)
        line += f" {_mean(rs, 'total_reward'):>10}"
        print(line)

    # --- Table 3: outcome rate by target slot ---
    by_slot: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_slot[r["slot_id"]].append(r)
    print("\n=== Outcome rate by target slot (0=Y+0.3 ... 3=Y-0.3) ===")
    print(
        f"{'slot':<12} {'n_rows':>7} " + " ".join(f"{s:>10}" for s in STATE_NAMES)
        + f" {'mean_total':>10} {'mean_dist_xy':>12} {'mean_dist_z':>12}"
    )
    for sid in range(4):
        rs = by_slot.get(sid, [])
        n = len(rs)
        state_counts = [sum(1 for r in rs if r["state"] == s) for s in range(4)]
        line = f"{sid:<12} {n:>7} "
        line += " ".join(f"{_pct(c, n):>10}" for c in state_counts)
        line += f" {_mean(rs, 'total_reward'):>10} {_mean(rs, 'dist_xy'):>12} {_mean(rs, 'dist_z'):>12}"
        print(line)

    # --- Table 4: outcome rate by neighbor clearance ---
    # Only 1 of 4 slots is empty per episode, so a middle slot (1 or 2) can never have both
    # neighbors empty at once -- use real, non-degenerate buckets instead of a naive
    # left x right cross product: for middle slots, "has any empty neighbor" vs "boxed in on
    # both sides"; edge slots (0/3) only ever have one neighbor, bucketed separately.
    neighbor_buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        s = r["slot_id"]
        left, right = r["left_occupied"], r["right_occupied"]
        if s in (1, 2):
            if left == 0 or right == 0:
                neighbor_buckets["middle: has empty neighbor"].append(r)
            else:
                neighbor_buckets["middle: boxed in (both occupied)"].append(r)
        else:
            neighbor = right if s == 0 else left
            if neighbor == 0:
                neighbor_buckets["edge slot: neighbor empty"].append(r)
            else:
                neighbor_buckets["edge slot: neighbor occupied"].append(r)
    # n_rows/pct is step-weighted: a config the policy gets stuck in (never triggers NEXT,
    # never falls off) piles up rows for the rest of that episode, so its row share can look
    # very different from how often it's actually assigned. n_assign/assign_pct instead counts
    # each fresh command latch (mid-episode NEXT resample or reset) exactly once via
    # `is_new_assignment`, which is comparable directly against the theoretical assignment-time
    # split (33.3% / 16.7% / 16.7% / 33.3% for a uniform 3-cubes-in-4-slots layout, verified
    # against real mdp/commands.py + mdp/events.py logic via Monte Carlo, 2026-09-01). done_rate
    # is the fraction of this bucket's ROWS that are terminal (time_out or cube_off_table):
    # unusually high here vs. other buckets would support "this config's episodes end early"
    # as the reason its row share undershoots its assignment share, rather than dwell time.
    total_assignments = sum(1 for r in rows if r["is_new_assignment"])
    print("\n=== Outcome rate by neighbor clearance (\"more room to maneuver\" hypothesis) ===")
    header = (
        f"{'bucket':<32} {'n_rows':>7} {'pct':>6} {'n_assign':>8} {'assign_pct':>10} {'done_rate':>9} "
        + " ".join(f"{c:>8}" for c in _REWARD_COLS)
    )
    print(header)
    for key in (
        "middle: has empty neighbor", "middle: boxed in (both occupied)",
        "edge slot: neighbor empty", "edge slot: neighbor occupied",
    ):
        rs = neighbor_buckets.get(key, [])
        n = len(rs)
        n_assign = sum(1 for r in rs if r["is_new_assignment"])
        done_rate = sum(1 for r in rs if r["done"]) / n if n else 0.0
        line = f"{key:<32} {n:>7} {_pct(n, total_n)} {n_assign:>8} {_pct(n_assign, total_assignments):>10} {done_rate:>9.3f} "
        line += " ".join(_mean(rs, c) for c in _REWARD_COLS)
        print(line)


def write_csv(rows: list[dict], path: Path) -> None:
    """Raw per-(step,env) dump. size_id/color_id/state kept as ints; decode via
    SIZE_NAMES/COLOR_NAMES/STATE_NAMES above."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
