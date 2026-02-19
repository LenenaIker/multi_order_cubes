# multi_order_cubes/mdp/events.py
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch

from .commands import sample_command_from_to

def sample_from_to_on_reset(env: "ManagerBasedRLEnv", env_ids=None):
    """
    EventManager calls reset events as: func(env, env_ids, **params)
    Respect env_ids for partial resets.
    """
    if env_ids is None:
        sample_command_from_to(env)
    else:
        sample_command_from_to(env, env_ids=env_ids)


def randomize_cubes_on_slots(env, env_ids):
    device = env.device
    env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    # slots en frame local del env (cfg) -> world por env con env_origins
    slots_local = torch.as_tensor(env.cfg.slot_positions, device=device, dtype=torch.float32)  # (4,3)
    origins = env.scene.env_origins.index_select(0, env_ids)                                   # (M,3)
    slots_w = origins.unsqueeze(1) + slots_local.unsqueeze(0)                                  # (M,4,3)

    M = env_ids.numel()
    num_slots = 4
    num_cubes = 3

    # 3 slots sin repetición por env (batched)
    perm = torch.rand((M, num_slots), device=device).argsort(dim=1)[:, :num_cubes]            # (M,3)
    pos = slots_w.gather(1, perm.unsqueeze(-1).expand(-1, -1, 3))                               # (M,3,3)

    quat = torch.zeros((M, 4), device=device, dtype=torch.float32)
    quat[:, 0] = 1.0

    cubes = [env.scene["cube_1"], env.scene["cube_2"], env.scene["cube_3"]]
    zero_vel = torch.zeros((M, 6), device=device, dtype=torch.float32)

    for c, cube in enumerate(cubes):
        pose_c = torch.cat([pos[:, c, :], quat], dim=1)  # (M,7)
        cube.write_root_pose_to_sim(pose_c, env_ids=env_ids)
        cube.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)  # recomendado en reset

