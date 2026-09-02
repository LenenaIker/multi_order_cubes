from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.sim.schemas import activate_contact_sensors

from .commands import latch_target_cube_from_command, sample_command_from_to
from .constants import CUBE_KEYS_9
from .step_cache import get_active_cube_pos_w, invalidate_moc_cache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_COLOR_BASE = torch.tensor([0, 3, 6], dtype=torch.long)

_FINGER_LOCAL_PATHS = (
    "Robot/ee_link/left_inner_finger",
    "Robot/ee_link/right_inner_finger",
)


def activate_finger_contact_sensors(env: "ManagerBasedRLEnv", env_ids) -> None:
    """Prestartup event: apply PhysxContactReportAPI to the two gripper finger bodies.

    Scoped per finger prim path, never the robot root -- applying it at the robot
    root walks every rigid-body prim under the robot and crashes on `base_link`
    (the fixed articulation root, which lacks the PhysX attribute the walker
    expects). Runs after the scene is spawned but before `sim.reset()` initializes
    the ContactSensor (the "prestartup" event window), which is the required order.
    """
    for env_prim_path in env.scene.env_prim_paths:
        for local_path in _FINGER_LOCAL_PATHS:
            activate_contact_sensors(f"{env_prim_path}/{local_path}", threshold=1.0)


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


def moc_reset_on_reset(env: "ManagerBasedRLEnv", env_ids=None) -> None:
    if not hasattr(env, "_moc_reset_id") or env._moc_reset_id is None:
        env._moc_reset_id = 0
    env._moc_reset_id += 1

    randomize_cubes_on_slots(env, env_ids)

    try:
        if hasattr(env.scene, "write_data_to_sim"):
            env.scene.write_data_to_sim()
        if hasattr(env, "sim") and hasattr(env.sim, "step"):
            # render=False: this settle step runs on every reset (including mid-training,
            # headless runs) and previously always rendered because SimulationContext.step's
            # `render` param defaults to True.
            env.sim.step(render=False)
        if hasattr(env.scene, "update") and hasattr(env, "sim") and hasattr(env.sim, "get_physics_dt"):
            try:
                # InteractiveScene.update requires a `dt` argument; calling it with none (as
                # this used to) raised TypeError on every reset and was silently swallowed by
                # the except below, so asset data buffers were never actually refreshed after
                # the settle step above.
                env.scene.update(env.sim.get_physics_dt())
            except TypeError:
                pass
    except Exception:
        pass

    if env_ids is None:
        settle_env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        settle_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

    # randomize_cubes_on_slots recorded moc_cube_home_pos_w from the COMMANDED slot pose
    # (slot_positions' fixed z=0.021), which only matches the medium cube's actual resting
    # height. The small/large variants settle a few mm below/above that after the physics
    # step above (their collision half-height differs from the slot's calibrated z), which
    # silently biased reward_object_lifted by cube size: audit 2026-09-01 found large cubes
    # earning a permanent, unearned lift reward from the moment they spawn, and small cubes
    # needing extra unrewarded lift before object_lifted starts paying at all. Overwrite with
    # the pose actually settled by physics instead.
    if (
        settle_env_ids.numel() > 0
        and hasattr(env, "moc_cube_home_pos_w")
        and env.moc_cube_home_pos_w is not None
    ):
        invalidate_moc_cache(env)
        env.moc_cube_home_pos_w[settle_env_ids] = get_active_cube_pos_w(env)[settle_env_ids]

    # moc_next_cooldown/moc_next_signal previously survived across an episode reset: an env
    # that pressed NEXT right before timing out began its next episode still under the old
    # episode's cooldown, and that cooldown was fed straight into the policy's observation
    # (next_cooldown_obs) as if it belonged to the new episode.
    if hasattr(env, "moc_next_cooldown") and env.moc_next_cooldown is not None:
        env.moc_next_cooldown[settle_env_ids] = 0
    if hasattr(env, "moc_next_signal") and env.moc_next_signal is not None:
        env.moc_next_signal[settle_env_ids] = 0.0

    # Prioritized slot sampling (2026-09-02, DISABLED 2026-09-02, see mdp/commands.py): this
    # block only fed the disabled EMA-weighted sampling above. Left commented out rather than
    # deleted in case that feature is revisited later.
    # if (
    #     hasattr(env, "command_from_to") and env.command_from_to is not None
    #     and hasattr(env, "moc_command_ever_success") and env.moc_command_ever_success is not None
    # ):
    #     prev_from = env.command_from_to.index_select(0, settle_env_ids)[:, 0] - 1
    #     valid = prev_from >= 0
    #     if valid.any():
    #         ever_success = env.moc_command_ever_success.index_select(0, settle_env_ids)
    #         update_slot_success_ema(env, prev_from[valid], ever_success[valid])

    if not hasattr(env, "moc_command_ever_success") or env.moc_command_ever_success is None:
        env.moc_command_ever_success = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    env.moc_command_ever_success[settle_env_ids] = False

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

    A trigger alone does NOT resample: `env.moc_stable_success` (set by `reward_next_signal`
    this same step, same ordering guarantee as above) must also be true. Without this, NEXT
    was a free way to abandon a command the policy had failed to execute -- pay the small
    per-step penalty in `reward_next_signal` and get handed a brand new (from, to) pair,
    cooldown permitting, regardless of whether the old one was ever fulfilled. That let the
    policy learn to bail on hard commands instead of solving them (audit finding, 2026-09-01:
    slot1/slot2 collapsed to 0% grasp rate). A rejected press (triggered but not successful)
    does NOT consume the cooldown either -- it already sits at <=0 for the trigger to have
    fired, so leaving it alone (rather than resetting it) costs the policy nothing beyond the
    existing reward penalty, and it can try again next step for free.
    """
    if not hasattr(env, "moc_next_cooldown") or env.moc_next_cooldown is None:
        env.moc_next_cooldown = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

    if not hasattr(env, "moc_next_signal") or env.moc_next_signal is None:
        return

    trigger = next_trigger_mask(env, next_threshold=next_threshold, cooldown_steps=cooldown_steps).index_select(
        0, env_ids
    )

    if hasattr(env, "moc_stable_success") and env.moc_stable_success is not None:
        success = env.moc_stable_success.index_select(0, env_ids) > 0.5
    else:
        success = torch.zeros_like(trigger)

    # Prioritized slot sampling (2026-09-02, see mdp/commands.py): record "ever succeeded" every
    # step, not just on a trigger, so a command that reaches moc_stable_success without the
    # policy pressing NEXT that exact step still scores correctly if the episode ends before it
    # presses NEXT at all.
    if not hasattr(env, "moc_command_ever_success") or env.moc_command_ever_success is None:
        env.moc_command_ever_success = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    env.moc_command_ever_success[env_ids] |= success

    resample = trigger & success
    trigger_ids = env_ids[resample]
    idle_ids = env_ids[~resample]

    if idle_ids.numel() > 0:
        env.moc_next_cooldown[idle_ids] = (env.moc_next_cooldown.index_select(0, idle_ids) - 1).clamp(min=0)

    if trigger_ids.numel() > 0:
        # Prioritized slot sampling (2026-09-02, DISABLED 2026-09-02): only fed the disabled
        # EMA-weighted sampling in mdp/commands.py, left commented out rather than deleted.
        # prev_from = env.command_from_to.index_select(0, trigger_ids)[:, 0] - 1
        # update_slot_success_ema(env, prev_from, torch.ones_like(prev_from, dtype=torch.bool))

        sample_command_from_to(env, env_ids=trigger_ids)
        env.moc_next_cooldown[trigger_ids] = int(cooldown_steps)
        env.moc_command_ever_success[trigger_ids] = False