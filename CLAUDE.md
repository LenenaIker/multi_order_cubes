# multi_order_cubes (MOC)

RL environment built on **Isaac Lab / Isaac Sim** where a UR10e + Robotiq gripper learns to execute discrete
`(from_slot, to_slot)` cube-manipulation commands on a 4-slot table (3 cubes present per episode, 9 possible
cube variants: 3 colors × 3 sizes). Trained with Stable-Baselines3 SAC. Full README with project history and
motivation lives at `README.md` — read it for background, don't duplicate it here.

Intended pipeline: `Reach → Grasp → Lift → Transport → Place → NEXT`. Only **Reach** has been actively trained
and stabilized so far.

## Architecture rule — read before touching observations/rewards

This policy is a low-level **slave executor**. It receives a slot command and reacts to slot positions —
nothing more. The intended design has a higher-level planner (Cosmos Reason) that interprets *what* is in each
slot and issues commands.

**Never add object-semantic observations** (cube size, color, identity, etc.) to this policy's observation
space, even when they'd technically solve a real problem (e.g. grasp-aperture calibration, approach standoff).
That belongs to the planner, not this executor. If a physical calibration problem seems to need object
identity/size, the right direction is closed-loop/force-or-contact-based control or genuine sensory input
(vision) — not privileged simulator scalars fed as observations. This was rejected explicitly once already;
don't re-propose it.

## Launching Isaac Sim / Isaac Lab scripts — proceed with caution

**Do not execute** any command that touches the Isaac Sim runtime (sourcing `_isaac_sim/setup_conda_env.sh`,
`import isaacsim`/`omni.*` probes, launching `AppLauncher`-based scripts like `train_sb3_sac.py` / `play_sb3.py`)
without asking first and explaining exactly what will run. **Propose, let the user run it, then read the
results back.** This applies even if the command looks harmless/read-only from a technical standpoint.

Why: there's a real history (~5 months prior to this file, roughly a month of failed attempts) of "breaking"
the Isaac Sim install through launches whose root cause is no longer known — so unreviewed execution in this
domain is a genuine risk, not just caution theater.

Read-only investigation (reading files, `git log`, grepping shell history, `pip show`, `find`) is always fine.

### Verified working launch procedure (once approved to run)

```bash
conda activate env_isaaclab
source /home/lenena-iker/work/isaac/IsaacLab/_isaac_sim/setup_conda_env.sh
cd /home/lenena-iker/work/isaac/Learning-Isaac
PYTHONPATH=/home/lenena-iker/work/isaac/Learning-Isaac python multi_order_cubes/scripts/train_sb3_sac.py --num_envs <N> --total_timesteps <N> [other args]
```

Same pattern for `multi_order_cubes/scripts/play_sb3.py`. Two non-obvious reasons each piece is required:

- `env_isaaclab` was created via plain `conda create`, not `./isaaclab.sh --conda`, so it never got the
  `activate.d` hook that sources Isaac Sim's runtime — has to be sourced by hand every fresh terminal.
- `multi_order_cubes` isn't pip-installed and has no persisted `PYTHONPATH`; scripts are invoked by path
  (`python multi_order_cubes/scripts/...`), never via `-m`, so `PYTHONPATH=` has to carry the parent dir.

Output lands at `logs/sb3/multi_order_cubes_sac/<run_name>/` (relative to `Learning-Isaac/`): `tb/`,
`checkpoints/`, `final_sac.zip`, `vecnormalize.pkl`, `videos/`.

## Current reward/termination state (as of the last validated round)

Shipped and validated end-to-end (real Isaac Sim smoke test, 500/500 steps):
- `reward_next_signal` (`mdp/rewards.py`) — bonus/penalty for pressing NEXT in/out of the success zone.
- `cube_fell_off_table` termination (`mdp/terminations.py`) + `penalty_cube_off_table` reward.
- `penalty_bystander_displacement` — soft continuous penalty for displacing the 2 non-target cubes, excluding
  the current target. Exists specifically to avoid repeating a past failure mode: an earlier hard-terminate-only
  fall penalty made the agent afraid to approach cubes near the table edge at all.
- `slot_positions` deliberately unchanged (cubes stay away from the arm) — this fixed the arm's elbow bumping
  cubes; don't reverse it without the user asking.

Diagnosed but explicitly deferred — don't start on these without being asked again:
- `reach_z_gated`'s reward optimum sits inside the cube's own volume, not properly offset above it.
- The reach target is live-tracked each step rather than frozen at episode/command start.
- The target cube itself (unlike the 2 bystanders) has no protection from being pushed off-target before grasp.
- No phase/state-machine observation yet (`moc_phase_obs` stays a 0 placeholder) — planned for when
  Grasp/Lift phases exist.

## Working conventions

- Never run `git` commands (status, log, diff, add, commit, push, etc.) unless the user explicitly asks for it
  in that turn.
- Never add yourself as a co-author (e.g. `Co-Authored-By: Claude ...`) on any commit, push, or other change.
- Avoid em dashes in comments, docs, and other written content. Use a comma, parentheses, or a separate
  sentence instead.

## Layout

- `moc_env_cfg.py` — environment config (scene, actions, observations, rewards, terminations wiring).
- `mdp/` — `commands.py`, `events.py`, `observations.py`, `rewards.py`, `terminations.py`, `next_flag_action.py`,
  `constants.py`, `step_cache.py`.
- `scripts/train_sb3_sac.py`, `scripts/play_sb3.py` — SB3 SAC training/rollout entry points.
- `cfg/sb3_sac.yaml` — SB3 hyperparameters.
- `config/ur10_gripper/` — robot articulation config.
- `PhysicsInspector.py` — standalone physics debugging tool.
