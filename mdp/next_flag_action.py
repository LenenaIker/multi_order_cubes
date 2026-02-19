# multi_order_cubes/mdp/next_flag_action.py
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
        """
        Called by ActionManager every step with the slice of the full action vector
        corresponding to this term.
        """
        a = actions.view(-1, 1).to(torch.float32)
        self._raw[:] = a

        # No special processing; clamp to [-1, 1] for safety
        self._processed[:] = torch.clamp(a, -1.0, 1.0)

    def apply_actions(self) -> None:
        """
        Called after process_actions. Applies the processed actions to the environment.
        Here we just write a scalar per env.
        """
        self._env.moc_next_signal = self._processed.view(-1)


@configclass
class NextFlagActionCfg(ActionTermCfg):
    class_type: type = NextFlagAction
    asset_name: str = "robot"  # required by cfg.validate()
