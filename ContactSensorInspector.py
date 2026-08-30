"""Standalone diagnostic tool for the gripper finger contact sensors.

Step 1 (done, kept for reference): a prim-tree dump found the real finger paths are one
level deeper than the previous Grasp-Lift attempt assumed:
    {ENV_REGEX_NS}/Robot/ee_link/left_inner_finger   (not .../Robot/left_inner_finger)
    {ENV_REGEX_NS}/Robot/ee_link/right_inner_finger
That wrong path is exactly what made `ContactSensor.__init__` raise a RuntimeError inside
`ManagerBasedRLEnv.__init__` last time (see .grasp_lift_backup_20260826_234010/).

Step 2 (done): constructing a real `ContactSensorCfg` at the corrected path still crashed,
this time with "could not find any bodies with contact reporter API" -- PhysX needs
`PhysxContactReportAPI` applied to the finger bodies *before* `ContactSensor._initialize_impl()`
runs (inside `ManagerBasedRLEnv.__init__` -> `self.sim.reset()`).

Step 3 (tried, failed differently): mutating `env_cfg.scene.robot.spawn.rigid_props.
activate_contact_sensors = True` toggles the whole robot articulation at spawn time. This
walks every prim under the robot root with a rigid-body schema and crashed on `base_link`
("Attribute 'ActivateContactSensors' does not exist on prim '.../Robot/base_link'"), the
fixed/immobile articulation root. Dead end as written -- too broad a scope.

Step 4 (this run): scope the activation to exactly the two finger prims, using the sanctioned
gap `ManagerBasedEnv.__init__` already has for this. Reading isaaclab's own
`manager_based_env.py`: `self.scene = InteractiveScene(...)` runs, THEN (if
"prestartup" in event_manager.available_modes) `event_manager.apply(mode="prestartup")` runs,
THEN `self.sim.reset()` runs (which is what triggers ContactSensor init). So a plain
`EventTermCfg(mode="prestartup")` event term fires in exactly the gap we need, with the robot
already spawned (finger prims exist) and physics not yet started. No black-box
reconstruction of `ManagerBasedRLEnv.__init__` needed. Its one precondition,
`scene.cfg.replicate_physics == False`, is already true for this project's scene cfg.

Inside that event term we call `isaaclab.sim.schemas.activate_contact_sensors(prim_path, ...)`
directly on each finger's own prim path (not the robot root), once per env. That function
only walks the subtree under the given path, so it can never touch `base_link`.

Single env, CPU physics, no training loop: the same safe pattern as PhysicsInspector.py.

Step 5 (this run): the construction recipe from Step 4 is confirmed working (both finger
sensors report non-empty body_names, see salida_consola.txt from the prior run -- if the
contact report API had not been active in time, ContactSensor._initialize_impl() would have
raised the Step 2 RuntimeError instead of returning a body). What's not yet verified is that
the sensor reports a real nonzero force during an actual contact. This run switches the tail
of the script from a passive render-only loop to an active physics-stepping loop (same
`env.sim.step()` + `env.scene.update(physics_dt)` pattern PhysicsInspector.py already uses),
reads `net_forces_w` off both finger contact sensors every step, keeps an all-time running max
per finger, and prints the running max (and current instantaneous reading) about once per
sim-second. Intent: leave this running, manually perturb the fingers/cubes via the Isaac Sim
GUI (drag a cube into a finger, or similar), and watch the printed max go from 0 to nonzero.

Step 6 (this run, follow-up): Step 5's real-world test (via the Physics Inspector's joint
drive sliders, Tools > Physics > Physics Inspector) hit table contact fine (left finger
net_max=157.07 from bumping the table) but read exactly net_now=0.0000 on BOTH fingers while
visibly gripping a cube with both. `force_threshold` was ruled out as the cause -- it only
gates the derived `is_contact` boolean and the debug-marker color (contact_sensor.py:415),
never the raw `net_forces_w`/`force_matrix_w` tensors. Added `force_matrix_w` tracking
(filtered specifically against the 9 cube prims via `filter_prim_paths_expr`, already
configured) alongside `net_forces_w`, to tell "no contact reported at all against cubes" apart
from a sampling/timing fluke.

Step 7 (this run): re-tested with force_matrix_w added. Result: net_forces_w on BOTH fingers
now reads genuine live values while gripping (left ~3.7, right ~7.0, tracking the actual
squeeze), so contact IS being detected and attributed to the correct finger bodies -- ruling
out the earlier "instanceable Fingertip_01/Finger4_01 mesh" suspicion from the Step 1 prim
dump. But force_matrix_w (the cube-filtered channel) stayed exactly 0.0000 the whole time.
The PhysX tensor API docs for `create_rigid_contact_view`
(omni.physics.tensors impl/api.py:390-392) say filter targets "will not report their contacts
directly, but only when they are in contact with the sensors their contact forces will be
included" -- implying the cube itself shouldn't need PhysxContactReportAPI for this to work,
and a matching reference task (isaaclab_tasks/manager_based/manipulation/dexsuite) filters
fingers against an `Object` whose rigid_props also never sets `activate_contact_sensors`.
That contradicts our own measurement though. Since it's cheap and carries none of the Step 3
base_link risk (cubes are plain RigidObjects, not part of the robot articulation, so no
recursive walk that could hit an incompatible prim), this run also activates
PhysxContactReportAPI on all 9 cube prims from the same prestartup event, as a direct
empirical test of whether the filter target needs the API in practice, docs notwithstanding.

Step 8 (this run): re-tested with cubes activated too. Result: the Step 7 hypothesis is
FALSIFIED -- cube_max stayed exactly 0.0000 for both fingers throughout a sustained,
progressively-harder grip, even though net_max climbed the whole time (left up to 11.87,
right up to 13.07), so real contact force was present and growing but force_matrix_w never
picked any of it up. Activating PhysxContactReportAPI on the cube was not the missing piece.
force_matrix_w is now an open dead end; not investigated further this session in favor of the
practical path below.

Also observed (not a bug): `net_now` prints 0.0000 almost every time despite `net_max`
climbing, even mid-grip. This is a display/sampling artifact, not a computation error -- the
loop prints once per ~1 sim-second (`print_every_n_steps`), so `net_now` is a single
instantaneous physics-step sample, while `net_max` is updated from EVERY physics step in
between prints. PhysX contact force readings are naturally spiky/noisy at the single-step
level, so the printed instant frequently lands on a near-zero sample while the true peaks
(correctly) only show up in `net_max`.

**Practical conclusion for this investigation:** `net_forces_w` per finger (generic contact
force, not filtered to any specific object) is a working, real signal -- confirmed responding
live to an actual progressive grip. `force_matrix_w` (which body specifically) stays broken
for reasons not yet understood, but per this project's own architecture rule (never feed this
policy object identity -- see CLAUDE.md), a generic per-finger contact-force reading is
actually the *right* signal to want anyway: it's the closed-loop/force-based signal the
architecture doc explicitly points to as the correct alternative to privileged object
observations, not a workaround. Recommend treating `net_forces_w` as sufficient and moving on
to porting this recipe into the real mdp/ code, rather than continuing to chase
force_matrix_w.

Step 9 (this run, follow-up): user pushback on the Step 7/8 "sampling artifact" explanation,
correctly -- net_now read 0.0000 on every single print across a continuous, progressively
harder grip that never let go, which is a lot of consecutive unlucky single-step samples if
it really were just sampling noise. The loop code itself has no bug (current_net/current_cube
were already recomputed every physics step, not just on the print step), so the raw
per-physics-step `net_forces_w` reading really can be that close to zero most individual
steps even under sustained load -- plausible for a settled/static contact resolved mostly by
the joint drive's position correction rather than continuous contact-solver impulses, but not
confirmed, and not something to keep guessing about from code reading alone. Replaced the
single noisy instantaneous sample with real window statistics, reset every print interval
(~1 sim-second): `*_avg` (mean over the window -- should track a sustained grip if the true
force is in fact continuously nonzero), `*_peak` (max within just that window), and
`*_allmax` (the old never-reset running max, kept for reference). If `*_avg` also reads ~0
during a held grip, that would mean the raw signal genuinely spends most physics steps near
zero even while contact visibly persists -- useful to know either way for reward-shaping
later, since it would mean any real usage needs a peak/max-based signal, not a raw
instantaneous reading.

Step 10 (this run, follow-up): user re-tested playing squeeze/release (not a held grip) and
the window stats tracked it cleanly -- exactly 0 during release, nonzero and sensible during
squeeze. But then user noticed something more specific: holding the squeeze force *constant*
(not releasing, just not changing it further) also decays net_avg back to 0. Only a *changing*
force reads nonzero. This points at PhysX rigid-body sleeping: `RigidBodyPropertiesCfg.
sleep_threshold` defaults to `None`, which means "use PhysX's own nonzero default" -- a cube
held still under a steady grip loses kinetic energy and can drop below that threshold, putting
it to sleep. A sleeping body isn't actively re-solved for contacts each step (or its reported
contact data goes stale/zero), even though the physical contact is still real -- this would
exactly explain "changing force reads nonzero, constant force reads zero" and is the worst
possible failure mode for a grasp reward (it would falsely zero out exactly during a stable,
successful hold). Testing directly: forcing `sleep_threshold = 0.0` on all 9 cube RigidObjectCfg
spawns before construction, so none of them can ever sleep. Confirmed working in the real
training config port afterward (see mdp/events.py, config/ur10_gripper/moc_ur10_env_cfg.py).

Step 11 (this run): after porting the recipe into the real training config and training on
it, the policy's gripper spends most of its time closed (`grip_frac` up to ~0.6-0.78) with no
reward yet tied to contact force at all -- so this can't currently be a reward exploit through
our sensor. But before ever wiring a real grasp/contact reward, we need to know whether
squeezing the two fingers together with literally nothing between them produces any signal.
Reading `universal_robots.py` shows `enabled_self_collisions=False` on the robot articulation
(never overridden in this project), which should mean PhysX doesn't even generate a collision
pair between the robot's own links, so finger-vs-finger contact should be physically
impossible to detect. This run tests that live rather than trusting the config read alone: all
9 cubes are forced parked far away right after `env.reset()` (same "parked" offset
`randomize_cubes_on_slots` already uses for inactive cubes), guaranteeing the gripper's
workspace is empty, so whatever the sensors read while squeezing here can only be
finger-vs-finger, table, or robot self-contact -- never a cube.

Step 11 result: net_avg/net_peak on both fingers stayed exactly 0.0000 through a full
squeeze. Consistent with the `enabled_self_collisions=False` read, but that alone doesn't
distinguish "no signal because there genuinely is no contact" from "no signal even though
there IS real mechanical resistance" (which would be a much worse, separate bug -- a real
contact silently not being reported). A grep across mdp/rewards.py, mdp/observations.py,
mdp/events.py and moc_env_cfg.py's RewardsCfg also confirms no active (nonzero-weight) reward
term reads gripper/finger joint state at all; `diag_grip_distance` always `return
torch.zeros(...)`, so its contribution is `0 * weight * dt == 0` regardless of weight,
confirmed against RewardManager's own `value = term_cfg.func(...) * term_cfg.weight * dt`. So
whatever is driving `grip_frac` up in training is not a reward exploit through this reward or
through the contact sensor -- there is currently no path from "gripper closed" to "reward"
anywhere in the active reward stack.

Step 12 (this run): distinguish "genuinely no contact" from "contact but silently
unreported" by tracking the real geometric gap between the two finger bodies
(`robot.data.body_pos_w` for `left_inner_finger`/`right_inner_finger`) and the actuator's
`applied_torque` on `finger_joint` through the same squeeze. If the gap keeps shrinking
smoothly all the way to full closure with torque staying near its steady-state drive level
(no spike), that's direct mechanical evidence of frictionless pass-through, matching net=0
being correct rather than a missed detection. If the gap plateaus before full closure while
torque climbs, that would mean real resistance exists despite net_forces_w reading 0 -- a
genuine sensor bug, not an expected self-collision-off outcome.

Step 13 (this run): new investigation into the 20M-run reward-hacking finding (policy holds a
fake grasp by pressing both fingers against the table in perfect pre-grasp position, since
`get_finger_contact_force_w` in mdp/step_cache.py collapses `net_forces_w` to a magnitude via
`torch.linalg.norm(...)`, discarding direction -- a real cube pinch and a table press are
indistinguishable to a magnitude-only reading). Two candidate fixes were on the table:
filtering the sensor to cube prims only (`filter_prim_paths_expr` / `force_matrix_w`), and
using force *direction* instead of magnitude. The filter path is already a dead end per Step
6-8 above: `force_matrix_w` read exactly 0.0000 even while visibly gripping a cube with both
fingers, so it can't be trusted as a per-object channel. This run drops that dead branch
entirely and reverts the Step 11 "park every cube away" override (that was specifically for
the self-collision test, and blocks the workspace here), so a real cube sits reachable on a
slot after `env.reset()` like a normal episode. It then tracks the raw `net_forces_w` *vector*
(not just its norm) per finger, window-averaged the same way Step 9 introduced for magnitude,
plus the cosine similarity between the left and right average vectors. Hypothesis: pressing
both fingers against the table pushes both fingers in roughly the same direction (up, away
from the table -- cosine near +1), while pinching a cube between the fingers pushes them apart
from each other along the pinch axis (opposing -- cosine near -1). If that holds, the fix
needs zero changes to `ContactSensorCfg` in the real training config -- only `step_cache.py`'s
`get_finger_contact_force_w` would need to read direction instead of discarding it.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim.schemas import activate_contact_sensors  # noqa: E402
from multi_order_cubes.config.ur10_gripper import UR10LongSuctionMultiOrderCubesEnvCfg  # noqa: E402
from multi_order_cubes import mdp  # noqa: E402

FINGER_LOCAL_PATHS = [
    "Robot/ee_link/left_inner_finger",
    "Robot/ee_link/right_inner_finger",
]


def activate_finger_contact_sensors(env: "ManagerBasedRLEnv", env_ids) -> None:
    """Prestartup event: apply PhysxContactReportAPI to the two finger bodies AND the
    9 cube bodies (Step 6 experiment: cubes are plain RigidObjects, not part of the robot
    articulation, so activating them individually here carries none of the Step 3
    base_link risk).

    Scoped per body prim path (never the robot root), so this cannot repeat the
    Step 3 crash on `base_link`. Runs once, after everything is spawned but before
    `sim.reset()` activates physics handles -- see module docstring, Step 4.
    """
    for env_prim_path in env.scene.env_prim_paths:
        for local_path in FINGER_LOCAL_PATHS:
            prim_path = f"{env_prim_path}/{local_path}"
            print(f"[EVENT] activating contact sensor API on: {prim_path}")
            activate_contact_sensors(prim_path, threshold=1.0)

        for cube_key in mdp.CUBE_KEYS_9:
            prim_path = f"{env_prim_path}/{cube_key}"
            print(f"[EVENT] activating contact sensor API on: {prim_path}")
            activate_contact_sensors(prim_path, threshold=1.0)


def dump_prim_tree(stage, root_path: str, max_depth: int = 8) -> None:
    from pxr import UsdPhysics

    root_prim = stage.GetPrimAtPath(root_path)
    print(f"[DUMP] root={root_path} valid={root_prim.IsValid()}")
    if not root_prim.IsValid():
        return

    def _walk(prim, depth):
        if depth > max_depth:
            return
        for child in prim.GetChildren():
            name = child.GetPath().pathString
            print(
                f"[DUMP] {'  ' * depth}{name} "
                f"instance={child.IsInstance()} instance_proxy={child.IsInstanceProxy()} "
                f"rigid_body={child.HasAPI(UsdPhysics.RigidBodyAPI)}"
            )
            _walk(child, depth + 1)

    _walk(root_prim, 0)


def main():
    env_cfg = UR10LongSuctionMultiOrderCubesEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = "cpu"
    env_cfg.sim.physx.use_gpu = False

    # Step 10 experiment: sleep_threshold=None means "use PhysX's nonzero default", so a cube
    # held perfectly static under a steady grip can fall below it and go to sleep -- while
    # asleep, PhysX stops actively resolving/reporting that body's contact forces each step,
    # even though the physical contact is still there. Force sleep_threshold=0.0 on every cube
    # so none of them can ever sleep, as a direct test of the Step 9 finding (net_avg reads 0
    # under a held-constant squeeze, only nonzero while the force is actively changing).
    for cube_key in mdp.CUBE_KEYS_9:
        cube_cfg = getattr(env_cfg.scene, cube_key)
        cube_cfg.spawn.rigid_props.sleep_threshold = 0.0

    # Step 13: no filter_prim_paths_expr -- matches the real training config
    # (config/ur10_gripper/moc_ur10_env_cfg.py) exactly, since force_matrix_w is a
    # confirmed dead end (Step 6-8) and we're testing a direction-based signal instead,
    # which only needs the same unfiltered net_forces_w already used in production.
    env_cfg.scene.left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/left_inner_finger",
        force_threshold=1.0,
    )
    env_cfg.scene.right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/right_inner_finger",
        force_threshold=1.0,
    )

    env_cfg.events.moc_activate_finger_contacts = EventTermCfg(
        func=activate_finger_contact_sensors,
        mode="prestartup",
        params={},
    )

    print("=== Constructing ManagerBasedRLEnv with left/right_finger_contact attached... ===")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("=== ManagerBasedRLEnv constructed without crashing. ContactSensorCfg path is valid. ===")

    print("=== robot body_names (articulation view) ===")
    print(env.scene["robot"].body_names)
    print("=== left_finger_contact body_names:", env.scene["left_finger_contact"].body_names, "===")
    print("=== right_finger_contact body_names:", env.scene["right_finger_contact"].body_names, "===")

    env.reset()

    # Step 13: unlike Step 11 (self-collision test, needed an empty workspace), this test
    # needs a real cube in reach -- env.reset() already ran moc_reset_on_reset, which places
    # 3 of the 9 cubes onto real slot positions via randomize_cubes_on_slots (the other 6 get
    # parked env_origin+[5.0, y_off, 0.20], which is NOT distinguishable from a real slot by z
    # alone). Read the active-cube bookkeeping that event itself sets, and report where the
    # active cubes landed instead of re-parking everything.
    active_indices = env.active_cube_indices[0].tolist()
    for active_idx in active_indices:
        cube_key = mdp.CUBE_KEYS_9[active_idx]
        pos = env.scene[cube_key].data.root_pos_w[0]
        print(f"=== reachable cube: {cube_key} at world pos {pos.tolist()} ===")

    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    robot_path = f"{env.scene.env_prim_paths[0]}/Robot"
    dump_prim_tree(stage, robot_path)

    print(
        "=== Dump complete. Entering active physics loop. Using the Physics Inspector's "
        "joint sliders, try two separate things and watch the printed vectors: "
        "(1) drive the arm down and touch/press ONE OR BOTH fingers flat against the table "
        "(gripper can stay open or closed, doesn't matter), then "
        "(2) position the gripper over the reachable cube printed above and close the fingers "
        "to actually pinch/squeeze it. Watch left_vec/right_vec (avg force vector over the last "
        "~1s window) and cos_sim (angle between them): table-press is expected to push both "
        "fingers roughly the same direction (cos_sim near +1), a real cube pinch is expected "
        "to push them apart from each other (cos_sim near -1). ==="
    )

    physics_dt = env.cfg.sim.dt
    print_every_n_steps = max(1, int(round(1.0 / physics_dt)))  # ~once per sim-second

    # Step 12: geometric gap between the two finger bodies + drive torque on finger_joint,
    # to tell "genuinely no contact" apart from "contact but silently unreported".
    robot = env.scene["robot"]
    finger_body_ids, finger_body_names = robot.find_bodies(["left_inner_finger", "right_inner_finger"])
    finger_joint_ids, _ = robot.find_joints(["finger_joint"])
    finger_joint_id = finger_joint_ids[0]
    print(f"=== tracking body gap for {finger_body_names} (ids {finger_body_ids}) ===")

    sensor_names = ["left_finger_contact", "right_finger_contact"]
    running_max_net = {name: 0.0 for name in sensor_names}
    # window accumulators: reset every print interval, so "avg" reflects the last ~1
    # sim-second instead of one noisy single-step sample (see Step 9 in the module docstring).
    # Step 13: accumulate the full force *vector*, not just its magnitude, so direction
    # survives the windowed average instead of being discarded like step_cache.py does today.
    window_vec_sum = {name: torch.zeros(3) for name in sensor_names}
    window_peak_net = {name: 0.0 for name in sensor_names}
    window_steps = 0
    step_count = 0

    while simulation_app.is_running():
        env.sim.step()
        env.scene.update(physics_dt)

        for sensor_name in sensor_names:
            data = env.scene[sensor_name].data
            if data.net_forces_w is None:
                continue

            vec = data.net_forces_w[0, 0, :]  # (num_envs, num_bodies, 3) -> this env/body
            net_mag = torch.linalg.norm(vec).item()
            window_vec_sum[sensor_name] += vec
            window_peak_net[sensor_name] = max(window_peak_net[sensor_name], net_mag)
            running_max_net[sensor_name] = max(running_max_net[sensor_name], net_mag)

        window_steps += 1

        if step_count % print_every_n_steps == 0:
            finger_pos = robot.data.body_pos_w[0, finger_body_ids, :]
            finger_gap = torch.linalg.norm(finger_pos[0] - finger_pos[1]).item()
            closed_frac = (
                (robot.data.joint_pos[0, finger_joint_id] - robot.data.joint_pos_limits[0, finger_joint_id, 0])
                / (robot.data.joint_pos_limits[0, finger_joint_id, 1] - robot.data.joint_pos_limits[0, finger_joint_id, 0]).clamp(min=1e-6)
            ).item()
            finger_torque = robot.data.applied_torque[0, finger_joint_id].item()

            left_avg = window_vec_sum["left_finger_contact"] / max(window_steps, 1)
            right_avg = window_vec_sum["right_finger_contact"] / max(window_steps, 1)
            left_norm = torch.linalg.norm(left_avg).item()
            right_norm = torch.linalg.norm(right_avg).item()
            if left_norm > 1e-6 and right_norm > 1e-6:
                cos_sim = (torch.dot(left_avg, right_avg) / (left_norm * right_norm)).item()
                cos_sim_str = f"{cos_sim:+.3f}"
            else:
                cos_sim_str = "n/a (near-zero force)"

            print(
                f"[{step_count:06d}] gap={finger_gap:.4f} closed_frac={closed_frac:.3f} torque={finger_torque:+.4f}\n"
                f"    left_vec=({left_avg[0]:+.3f},{left_avg[1]:+.3f},{left_avg[2]:+.3f}) "
                f"|left|={left_norm:.4f} peak={window_peak_net['left_finger_contact']:.4f} allmax={running_max_net['left_finger_contact']:.4f}\n"
                f"    right_vec=({right_avg[0]:+.3f},{right_avg[1]:+.3f},{right_avg[2]:+.3f}) "
                f"|right|={right_norm:.4f} peak={window_peak_net['right_finger_contact']:.4f} allmax={running_max_net['right_finger_contact']:.4f}\n"
                f"    cos_sim(left,right)={cos_sim_str}"
            )
            window_vec_sum = {name: torch.zeros(3) for name in sensor_names}
            window_peak_net = {name: 0.0 for name in sensor_names}
            window_steps = 0

        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
