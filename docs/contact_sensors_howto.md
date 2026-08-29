# How to get PhysX contact sensors working in Isaac Lab

Tutorial-style writeup of a recipe that took a full debugging session to work out
(2026-08-28/29, this project). Written to be portable to any Isaac Lab project, not
just this one. A worked reference implementation lives in `ContactSensorInspector.py`
in this repo (standalone script, `num_envs=1`, never touches the real training config).

## The problem this solves

You want a `ContactSensor` reading real contact force off some rigid body (e.g. a
gripper finger) at runtime. Isaac Lab makes this look like "just add a
`ContactSensorCfg` to the scene," but the naive version crashes or silently reads
zero, for four separate reasons stacked on top of each other.

## Step 1 — get the prim path right

`ContactSensorCfg(prim_path=...)` needs the *exact* USD prim path of the body, and
robot USD hierarchies are often one or two levels deeper than you'd guess from the
articulation's `body_names`. Don't guess: dump the actual prim tree under the robot
root and match against it.

```python
import omni.usd
from pxr import Usd

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
for prim in Usd.PrimRange(root):
    print(prim.GetPath(), prim.GetTypeName())
```

In this project the fingers turned out to be nested under `ee_link`:
`{ENV_REGEX_NS}/Robot/ee_link/left_inner_finger`, not
`{ENV_REGEX_NS}/Robot/left_inner_finger` as a first guess assumed.

## Step 2 — the body needs `PhysxContactReportAPI` applied *before* the sensor initializes

PhysX doesn't track per-body contact reports by default; a rigid body needs the
`PhysxContactReportAPI` schema applied to it. `ContactSensor._initialize_impl()`
(called from inside `sim.reset()`) checks for this and raises if it's missing.

The obvious fix — flip `activate_contact_sensors=True` on the robot's spawn config
(`rigid_props`) — looks right but is a dead end:

```python
# DON'T do this on a whole articulation:
env_cfg.scene.robot.spawn.rigid_props.activate_contact_sensors = True
```

This walks *every* prim under the robot that has a rigid-body schema and tries to
apply the API to each one. It crashes on the articulation's fixed root link (e.g.
`base_link`), which doesn't carry the attribute the walker expects. Scope matters:
you want this applied to specific bodies, not the whole tree.

## Step 3 — activate it narrowly, and at the right moment, via a `prestartup` event

The fix is to call the schema function directly, scoped to just the body (or bodies)
you care about:

```python
import isaaclab.sim.schemas as schemas

def activate_finger_contact_sensors(env, env_ids=None):
    for finger in ("left_inner_finger", "right_inner_finger"):
        schemas.activate_contact_sensors(f"{{ENV_REGEX_NS}}/Robot/ee_link/{finger}")
```

Timing matters as much as scope. This has to run:
- **after** the robot is spawned (the prim must exist to apply a schema to it), but
- **before** `sim.reset()` runs (which is when `ContactSensor` checks for the API).

Isaac Lab already has a hook for exactly this window: events registered with
`mode="prestartup"` fire inside `ManagerBasedEnv.__init__`, after `InteractiveScene`
construction (robot spawned) and before `sim.reset()` (sensor init). Register it as a
normal `EventTermCfg`:

```python
from isaaclab.envs.mdp import EventTermCfg

prestartup_finger_contacts = EventTermCfg(
    func=activate_finger_contact_sensors,
    mode="prestartup",
)
```

No need to hand-reconstruct `InteractiveScene` or subclass the env. One precondition:
`scene.cfg.replicate_physics` must be `False` (it already is in most manipulation
setups that need per-env randomization).

## Step 4 — read the signal correctly: which channel, and at what granularity

Once the sensor exists, `ContactSensor.data` exposes (at least) two channels:

- **`net_forces_w`** — total contact force on the body, source-agnostic. Doesn't know
  *what* it's touching, just that it's in contact and how hard.
- **`force_matrix_w`** — force filtered specifically against a list of target prims
  (`filter_prim_paths_expr`), i.e. "how hard am I pressing against *this specific
  object*."

In this project `force_matrix_w` never worked (stayed exactly `0.0` even after
applying the contact-report API to the filter targets too, contradicting PhysX's own
docs). Root cause unresolved, not investigated further. `net_forces_w` worked
perfectly. If you don't strictly need per-object identification, prefer
`net_forces_w` — it's simpler and, if your policy shouldn't know object identity
anyway, it's the architecturally correct channel too, not just a fallback.

**Read it as a window statistic, not a single instantaneous sample.** PhysX's
per-physics-step contact force is spiky; a single `net_forces_w` read once every N
steps can land on a near-zero sample even during a real, sustained contact. Track a
short rolling average/peak (e.g. over the last decimation window) instead of trusting
one raw sample.

## Step 5 — the trap: PhysX puts still bodies to sleep, and that includes your contact signal

This is the subtlest one and the easiest to silently carry a bug from: PhysX puts a
rigid body to sleep (stops actively resolving/reporting it) once its kinetic energy
drops below `sleep_threshold` for a while. A body held in a **stable, unmoving grip**
is exactly the case that triggers this. Once asleep, `net_forces_w` reads 0 even
though the physical contact is completely real.

For a grasp signal, this is the worst possible failure mode: it silently *zeroes out
right when the grip is successful and steady*, i.e. exactly the state you want to
reward.

Fix: force the object to never sleep by setting its sleep threshold to zero in its
`RigidBodyPropertiesCfg`:

```python
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

rigid_props = RigidBodyPropertiesCfg(
    ...,
    sleep_threshold=0.0,
)
```

Apply this to any object whose contact force you plan to read while it's expected to
sit still under load (the graspable object, not necessarily the robot body itself).

## Checklist for porting into a real project

1. Find the real prim paths for the bodies you want to sense (Step 1).
2. Add `ContactSensorCfg` entries to the scene config, pointed at those paths.
3. Add a scoped `prestartup` `EventTermCfg` that calls
   `isaaclab.sim.schemas.activate_contact_sensors(...)` per body, never on the whole
   robot/articulation root (Steps 2-3).
4. Set `sleep_threshold=0.0` on any rigid object whose steady-state contact force
   you need to read (Step 5) — a one-line cfg field, easy to forget, breaks the
   signal in exactly the case that matters most.
5. Read `net_forces_w` as a windowed avg/peak, not a raw single-step sample (Step 4).
6. Prefer `net_forces_w` over `force_matrix_w` unless you've separately verified
   `force_matrix_w` works for your asset/PhysX version; don't assume it does.
