from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import omni.timeline
from isaaclab.sim.schemas import activate_contact_sensors as _activate_contact_sensors_on_prim

from .commands import latch_target_cube_from_command, sample_command_from_to
from .constants import CUBE_KEYS_9
from .step_cache import get_active_cube_pos_w, get_finger_cube_contact_force

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_COLOR_BASE = torch.tensor([0, 3, 6], dtype=torch.long)


def activate_finger_contact_sensors(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    force_threshold: float = 1.0,
) -> None:
    """Applies PhysX's contact-report API to each env's finger links.

    STATUS: diagnostic pass, not a confirmed fix. Three straight guesses at the right timing/
    target for this have each failed with the identical symptom (`left_inner_finger` reports as
    an invalid USD prim path), including hooking the Kit timeline's own `PLAY` event ahead of
    `ContactSensor`'s own `order=10` callback (see git history of this function for the ruled-out
    attempts) — so the actual cause is evidently not a Python/event-ordering timing issue at all.
    Two real possibilities remain that only a live prim-tree dump can distinguish: either (a)
    `left_inner_finger` is not a direct child of `.../Robot` (this robot's USD likely nests each
    link under its kinematic parent, so the true path could be several levels deeper), or (b) it
    sits inside an `instanceable` sub-tree, where a plain `GetPrimAtPath`/schema-`Apply()` doesn't
    work the same way `ContactSensor`'s own `find_matching_prims`-based lookup does.

    This call is now wrapped in try/except (so a failure here logs and moves on instead of
    crashing) and, on the same `PLAY` event, dumps the real child-by-child structure under
    `.../Robot` (`instance`/`instance_proxy`/`rigid_body` flags per prim) via `omni.log.warn`, so
    the actual next fix can be based on what the tree really looks like instead of another guess.
    The run will still hit `ContactSensor`'s own RuntimeError right after — that's expected this
    round; the `[MOC DIAG]` lines printed just before it are what this pass is for.
    """
    def _do_activate(event):
        import omni.log
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdPhysics

        stage = get_current_stage()
        robot_path = f"{env.scene.env_prim_paths[0]}/Robot"
        robot_prim = stage.GetPrimAtPath(robot_path)
        omni.log.warn(f"[MOC DIAG] {robot_path} valid={robot_prim.IsValid()}")
        if robot_prim.IsValid():
            def _dump(prim, depth):
                if depth > 8:
                    return
                for child in prim.GetChildren():
                    name = child.GetPath().pathString
                    omni.log.warn(
                        f"[MOC DIAG]  {'  ' * depth}{name} "
                        f"instance={child.IsInstance()} instance_proxy={child.IsInstanceProxy()} "
                        f"rigid_body={child.HasAPI(UsdPhysics.RigidBodyAPI)}"
                    )
                    _dump(child, depth + 1)
            _dump(robot_prim, 0)

        for env_prim_path in env.scene.env_prim_paths:
            for finger in ("left_inner_finger", "right_inner_finger"):
                prim_path = f"{env_prim_path}/Robot/{finger}"
                try:
                    _activate_contact_sensors_on_prim(prim_path, threshold=force_threshold)
                except Exception as exc:
                    omni.log.warn(f"[MOC DIAG] activate failed for {prim_path}: {exc}")

    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    # kept on env (not a local var) so the subscription isn't garbage-collected/unsubscribed
    # the moment this function returns.
    env._moc_finger_contact_play_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY),
        _do_activate,
    )


def _maybe_set_visibility(cube, visible: bool, env_ids: torch.Tensor) -> None:
    if hasattr(cube, "set_visibility"):
        try:
            cube.set_visibility(visible, env_ids=env_ids)
        except Exception:
            pass


def randomize_cubes_on_slots(env: "ManagerBasedRLEnv", env_ids) -> None:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return

    num_envs = env_ids.numel()
    num_slots = 4
    num_active_cubes = 3

    slots_local = torch.as_tensor(env.cfg.slot_positions, dtype=torch.float32, device=env.device)
    origins = env.scene.env_origins.index_select(0, env_ids)
    slots_w = origins.unsqueeze(1) + slots_local.unsqueeze(0)

    active_slot_idx = torch.rand((num_envs, num_slots), device=env.device).argsort(dim=1)[:, :num_active_cubes]
    active_cube_pos_w = slots_w.gather(1, active_slot_idx.unsqueeze(-1).expand(-1, -1, 3))

    if not hasattr(env, "moc_cube_home_pos_w") or env.moc_cube_home_pos_w is None:
        env.moc_cube_home_pos_w = torch.zeros((env.num_envs, 3, 3), dtype=torch.float32, device=env.device)
    env.moc_cube_home_pos_w[env_ids] = active_cube_pos_w

    quat_identity = torch.zeros((num_envs, 4), dtype=torch.float32, device=env.device)
    quat_identity[:, 0] = 1.0
    zero_vel = torch.zeros((num_envs, 6), dtype=torch.float32, device=env.device)

    size_perm = torch.rand((num_envs, 3), device=env.device).argsort(dim=1)
    active_cube_indices = _COLOR_BASE.to(env.device).view(1, 3) + size_perm

    if not hasattr(env, "active_cube_indices") or env.active_cube_indices is None:
        env.active_cube_indices = torch.zeros((env.num_envs, 3), dtype=torch.long, device=env.device)
    env.active_cube_indices[env_ids] = active_cube_indices

    if not hasattr(env, "moc_active_cube_slot_idx") or env.moc_active_cube_slot_idx is None:
        env.moc_active_cube_slot_idx = torch.zeros((env.num_envs, 3), dtype=torch.long, device=env.device)

    if not hasattr(env, "moc_slot_to_active_id") or env.moc_slot_to_active_id is None:
        env.moc_slot_to_active_id = -torch.ones((env.num_envs, 4), dtype=torch.long, device=env.device)

    env.moc_active_cube_slot_idx[env_ids] = active_slot_idx.to(torch.long)
    env.moc_slot_to_active_id[env_ids] = -1

    for active_id in range(3):
        slot_idx = active_slot_idx[:, active_id].to(torch.long)
        env.moc_slot_to_active_id[env_ids, slot_idx] = active_id

    if not hasattr(env, "target_cube_id") or env.target_cube_id is None:
        env.target_cube_id = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    env.target_cube_id[env_ids] = 0

    for cube_idx, cube_key in enumerate(CUBE_KEYS_9):
        cube = env.scene[cube_key]

        is_active = (active_cube_indices == cube_idx).any(dim=1)
        if is_active.any():
            active_env_ids = env_ids[is_active]
            which_col = (active_cube_indices[is_active] == cube_idx).to(torch.int64).argmax(dim=1)

            pose = torch.cat(
                [
                    active_cube_pos_w[is_active]
                    .gather(1, which_col.view(-1, 1, 1).expand(-1, 1, 3))
                    .squeeze(1),
                    quat_identity[is_active],
                ],
                dim=1,
            )

            cube.write_root_pose_to_sim(pose, env_ids=active_env_ids)
            cube.write_root_velocity_to_sim(zero_vel[is_active], env_ids=active_env_ids)
            _maybe_set_visibility(cube, True, active_env_ids)

        is_inactive = ~is_active
        if is_inactive.any():
            inactive_env_ids = env_ids[is_inactive]
            y_off = (float(cube_idx) - 4.0) * 0.25
            parked_pos = origins[is_inactive] + torch.tensor([5.0, y_off, 0.20], dtype=torch.float32, device=env.device)
            parked_pose = torch.cat([parked_pos, quat_identity[is_inactive]], dim=1)

            cube.write_root_pose_to_sim(parked_pose, env_ids=inactive_env_ids)
            cube.write_root_velocity_to_sim(zero_vel[is_inactive], env_ids=inactive_env_ids)
            _maybe_set_visibility(cube, False, inactive_env_ids)


def _ensure_phase_buffers(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "moc_phase") or env.moc_phase is None:
        env.moc_phase = torch.ones((env.num_envs,), dtype=torch.long, device=env.device)
    if not hasattr(env, "moc_grasp_hold") or env.moc_grasp_hold is None:
        env.moc_grasp_hold = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    if not hasattr(env, "moc_lift_hold") or env.moc_lift_hold is None:
        env.moc_lift_hold = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    if not hasattr(env, "moc_grasp_trigger") or env.moc_grasp_trigger is None:
        env.moc_grasp_trigger = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    if not hasattr(env, "moc_lift_trigger") or env.moc_lift_trigger is None:
        env.moc_lift_trigger = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)


def moc_reset_on_reset(env: "ManagerBasedRLEnv", env_ids=None) -> None:
    if not hasattr(env, "_moc_reset_id") or env._moc_reset_id is None:
        env._moc_reset_id = 0
    env._moc_reset_id += 1

    randomize_cubes_on_slots(env, env_ids)

    _ensure_phase_buffers(env)
    reset_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(
        env_ids, dtype=torch.long, device=env.device
    )
    env.moc_phase[reset_ids] = 1
    env.moc_grasp_hold[reset_ids] = 0
    env.moc_lift_hold[reset_ids] = 0
    env.moc_grasp_trigger[reset_ids] = False
    env.moc_lift_trigger[reset_ids] = False

    try:
        if hasattr(env.scene, "write_data_to_sim"):
            env.scene.write_data_to_sim()
        if hasattr(env, "sim") and hasattr(env.sim, "step"):
            env.sim.step()
        if hasattr(env.scene, "update"):
            try:
                env.scene.update()
            except TypeError:
                pass
    except Exception:
        pass

    sample_command_from_to(env, env_ids=env_ids)
    latch_target_cube_from_command(env, env_ids)


def next_trigger_mask(
    env: "ManagerBasedRLEnv",
    next_threshold: float = 0.5,
    cooldown_steps: int = 30,
) -> torch.Tensor:
    """Boolean mask (num_envs,) of envs whose NEXT press would be consumed this step.

    Shared by `consume_next_signal` (which actually resamples the command) and
    `mdp.rewards.reward_next_signal` (which scores whether the press was correct), so both
    always agree on what counts as "pressing NEXT this step".
    """
    if not hasattr(env, "moc_next_cooldown") or env.moc_next_cooldown is None:
        env.moc_next_cooldown = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

    if not hasattr(env, "moc_next_signal") or env.moc_next_signal is None:
        return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    return (env.moc_next_signal > float(next_threshold)) & (env.moc_next_cooldown <= 0)


def consume_next_signal(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    next_threshold: float = 0.5,
    cooldown_steps: int = 30,
) -> None:
    """Consumes the policy's NEXT action every step.

    While only the Reach stage is trained, pressing NEXT does not end the episode: it
    just asks the environment for a new (from, to) command mid-episode, the same way an
    external planner (e.g. Cosmos Reason) is meant to hand out commands one at a time
    without the episode ending in between.

    The trigger mask is evaluated BEFORE the cooldown is touched, using the exact same
    cooldown values `reward_next_signal` already read earlier this step (reward_manager
    runs before interval events in `ManagerBasedRLEnv.step`, and nothing else mutates
    `moc_next_cooldown` in between). Decrementing first and checking after, like the old
    code did, made the event's own trigger disagree with the reward's for the specific
    step where the cooldown crosses from 1 to 0, silently un-rewarding that resample.
    """
    if not hasattr(env, "moc_next_cooldown") or env.moc_next_cooldown is None:
        env.moc_next_cooldown = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

    if not hasattr(env, "moc_next_signal") or env.moc_next_signal is None:
        return

    trigger = next_trigger_mask(env, next_threshold=next_threshold, cooldown_steps=cooldown_steps).index_select(
        0, env_ids
    )
    trigger_ids = env_ids[trigger]
    idle_ids = env_ids[~trigger]

    if idle_ids.numel() > 0:
        env.moc_next_cooldown[idle_ids] = (env.moc_next_cooldown.index_select(0, idle_ids) - 1).clamp(min=0)

    if trigger_ids.numel() > 0:
        sample_command_from_to(env, env_ids=trigger_ids)
        env.moc_next_cooldown[trigger_ids] = int(cooldown_steps)


def update_moc_phase(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    force_threshold: float = 1.0,
    grasp_hold_steps: int = 5,
    lift_height: float = 0.08,
    lift_hold_steps: int = 10,
    reach_exit_xy: float = 0.08,
    reach_exit_z: float = 0.05,
) -> None:
    """Advances/regresses env.moc_phase (1=Reach, 2=Grasp, 3=Lift) each step.

    Reads `env.moc_stable_success` and the `moc/reach_dist_xy` / `moc/reach_abs_dz` extras,
    both set earlier this same step by the reward manager (reward_next_signal / reward_reach_z_gated),
    the same before-event ordering `consume_next_signal` already relies on for moc_next_cooldown.

    reach_exit_xy/reach_exit_z are deliberately looser than reward_next_signal's success_xy/
    success_z, so there's a hysteresis band between "good enough to enter Grasp" and "bad enough
    to fall back to Reach" instead of chattering right at the boundary.
    """
    _ensure_phase_buffers(env)

    phase = env.moc_phase.index_select(0, env_ids)
    grasp_hold = env.moc_grasp_hold.index_select(0, env_ids)
    lift_hold = env.moc_lift_hold.index_select(0, env_ids)

    if hasattr(env, "moc_stable_success") and env.moc_stable_success is not None:
        reach_ok = env.moc_stable_success.index_select(0, env_ids).to(torch.bool)
    else:
        reach_ok = torch.zeros_like(phase, dtype=torch.bool)

    extras = env.extras if hasattr(env, "extras") and env.extras else {}
    dist_xy = extras.get("moc/reach_dist_xy")
    abs_dz = extras.get("moc/reach_abs_dz")
    if dist_xy is not None and abs_dz is not None:
        reach_lost = (dist_xy.index_select(0, env_ids) > float(reach_exit_xy)) | (
            abs_dz.index_select(0, env_ids) > float(reach_exit_z)
        )
    else:
        reach_lost = torch.zeros_like(phase, dtype=torch.bool)

    contact = get_finger_cube_contact_force(env).index_select(0, env_ids)
    grasp_ok = (contact[:, 0] > float(force_threshold)) & (contact[:, 1] > float(force_threshold))
    grasp_hold_new = torch.where(grasp_ok, grasp_hold + 1, torch.zeros_like(grasp_hold))

    cubes_now = get_active_cube_pos_w(env).index_select(0, env_ids)
    target_id = env.target_cube_id.index_select(0, env_ids).to(torch.long)
    row = torch.arange(env_ids.numel(), device=env.device)
    target_z_now = cubes_now[row, target_id, 2]
    target_z_home = env.moc_cube_home_pos_w.index_select(0, env_ids)[row, target_id, 2]

    lift_ok = grasp_ok & ((target_z_now - target_z_home) > float(lift_height))
    lift_hold_new = torch.where(lift_ok, lift_hold + 1, torch.zeros_like(lift_hold))

    grasp_trigger = (phase == 2) & (grasp_hold_new >= int(grasp_hold_steps)) & (grasp_hold < int(grasp_hold_steps))
    lift_trigger = (phase == 3) & (lift_hold_new >= int(lift_hold_steps)) & (lift_hold < int(lift_hold_steps))

    new_phase = phase.clone()
    new_phase = torch.where((new_phase == 1) & reach_ok, torch.full_like(new_phase, 2), new_phase)
    new_phase = torch.where(
        (new_phase == 2) & (grasp_hold_new >= int(grasp_hold_steps)), torch.full_like(new_phase, 3), new_phase
    )
    new_phase = torch.where((new_phase == 2) & reach_lost, torch.full_like(new_phase, 1), new_phase)
    new_phase = torch.where((new_phase == 3) & ~grasp_ok, torch.full_like(new_phase, 2), new_phase)

    env.moc_phase[env_ids] = new_phase
    env.moc_grasp_hold[env_ids] = grasp_hold_new
    env.moc_lift_hold[env_ids] = lift_hold_new
    env.moc_grasp_trigger[env_ids] = grasp_trigger
    env.moc_lift_trigger[env_ids] = lift_trigger