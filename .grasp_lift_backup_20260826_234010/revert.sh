#!/usr/bin/env bash
# Revierte SOLO los cambios de Grasp-Lift (contact sensors, phase state machine, rewards de
# grasp/lift), dejando intacto todo lo demás de la sesión (reach_z=9.5, cube_off_table, el
# print de body_names en PhysicsInspector.py, etc).
#
# Uso: desde la raíz de multi_order_cubes/
#   bash .grasp_lift_backup_20260826_234010/revert.sh
set -euo pipefail
BACKUP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BACKUP_DIR/.."

cp "$BACKUP_DIR/pre/moc_env_cfg.py" moc_env_cfg.py
cp "$BACKUP_DIR/pre/events.py" mdp/events.py
cp "$BACKUP_DIR/pre/rewards.py" mdp/rewards.py
cp "$BACKUP_DIR/pre/step_cache.py" mdp/step_cache.py
cp "$BACKUP_DIR/pre/moc_ur10_env_cfg.py" config/ur10_gripper/moc_ur10_env_cfg.py

echo "Revertido: moc_env_cfg.py, mdp/events.py, mdp/rewards.py, mdp/step_cache.py, config/ur10_gripper/moc_ur10_env_cfg.py"
echo "(el print de body_names en PhysicsInspector.py y el resto de cambios de la sesion NO se han tocado)"
