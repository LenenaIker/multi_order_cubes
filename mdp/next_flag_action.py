from __future__ import annotations

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class NextFlagAction(ActionTerm):
    """Stores the scalar NEXT action in env.moc_next_signal."""

    def __init__(self, cfg: "NextFlagActionCfg", env):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
        self._processed = torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = actions.view(-1, 1).to(torch.float32)
        self._raw[:] = actions
        self._processed[:] = torch.clamp(actions, -1.0, 1.0)

    def apply_actions(self) -> None:
        self._env.moc_next_signal = self._processed.view(-1)


@configclass
class NextFlagActionCfg(ActionTermCfg):
    class_type: type = NextFlagAction
    asset_name: str = "robot"