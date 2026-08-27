# Backup: cambios de Grasp-Lift (2026-08-26 23:40)

Copia de seguridad de los 5 archivos tocados al implementar la fase Grasp-Lift (contact
sensors dedo-cubo, `env.moc_phase` state machine, rewards `grasp_contact`/`lift_height`,
gating de `reach_xy`/`reach_z` por fase). Ver el plan completo en
`~/.claude/plans/lucky-painting-plum.md`.

## Contenido

- `pre/`  — los 5 archivos tal como estaban justo ANTES de esta implementación (con
  `reach_z=9.5` y todo lo demás de la sesión ya aplicado, pero sin nada de Grasp-Lift).
- `post/` — los 5 archivos tal como quedaron DESPUÉS (el estado que hay ahora mismo en el repo
  en el momento de este backup).
- `grasp_lift.diff` — diff unificado combinado, solo para inspección visual.
- `revert.sh` — copia `pre/` sobre los archivos reales del repo.

Archivos afectados:
- `moc_env_cfg.py`
- `mdp/events.py`
- `mdp/rewards.py`
- `mdp/step_cache.py`
- `config/ur10_gripper/moc_ur10_env_cfg.py`

## Si la run sale mal y quieres volver atrás

```bash
cd /home/lenena-iker/work/isaac/Learning-Isaac/multi_order_cubes
bash .grasp_lift_backup_20260826_234010/revert.sh
```

Esto deja el código exactamente como estaba antes de Grasp-Lift (Reach-only, `reach_z=9.5`,
sin contact sensors ni fase). No toca nada más (ni git, ni PhysicsInspector.py, ni cfg/sb3_sac.yaml).

## Si quieres volver a aplicar Grasp-Lift tras revertir

```bash
cd /home/lenena-iker/work/isaac/Learning-Isaac/multi_order_cubes
cp .grasp_lift_backup_20260826_234010/post/moc_env_cfg.py moc_env_cfg.py
cp .grasp_lift_backup_20260826_234010/post/events.py mdp/events.py
cp .grasp_lift_backup_20260826_234010/post/rewards.py mdp/rewards.py
cp .grasp_lift_backup_20260826_234010/post/step_cache.py mdp/step_cache.py
cp .grasp_lift_backup_20260826_234010/post/moc_ur10_env_cfg.py config/ur10_gripper/moc_ur10_env_cfg.py
```

Esta carpeta no está trackeada en git y no se ha commiteado nada. Puedes borrarla cuando ya no
la necesites.
