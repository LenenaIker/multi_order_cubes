from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg


class NextFlagAction(ActionTerm):
    """Stores continuous NEXT signal into env.moc_next_signal. No physics actuation."""

    def __init__(self, cfg: "NextFlagActionCfg", env):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
        self._processed = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

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
        a = actions.view(-1, 1).to(torch.float32)
        self._raw[:] = a
        self._processed[:] = torch.clamp(a, -1.0, 1.0)

    def apply_actions(self) -> None:
        """Write NEXT signal into env.moc_next_signal (no physics actuation)."""
        self._env.moc_next_signal = self._processed.view(-1)  # (num_envs,)


@configclass
class NextFlagActionCfg(ActionTermCfg):
    class_type: type = NextFlagAction
    asset_name: str = "robot"  # required by cfg.validate()