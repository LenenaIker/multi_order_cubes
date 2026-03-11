from __future__ import annotations

import torch


def time_out(env) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        raise AttributeError("env.episode_length_buf not found.")
    if not hasattr(env, "max_episode_length"):
        raise AttributeError("env.max_episode_length not found.")
    return env.episode_length_buf >= (int(env.max_episode_length) - 1)