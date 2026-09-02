"""Policy and critic networks for the skrl SAC agent.

skrl's `GaussianMixin.act()` samples from an *unbounded* Normal(mean, exp(log_std)) and, if
`clip_actions=True`, only hard-clamps the sample to the action space -- it does not apply the
tanh-squash + log-prob Jacobian correction that canonical SAC (Haarnoja et al. 2018, appendix C)
uses, and that SB3's `SAC` actor already validated on this project. `Policy.act()` below
reimplements that squash on top of `GaussianMixin.act()`'s raw sample.

Squashing alone is not enough: skrl's `IsaacLabWrapper.step()` (skrl/envs/wrappers/torch/
isaaclab_envs.py) passes actions straight through to `env.step()` with no rescale, whereas SB3's
`Sb3VecEnvWrapper` linearly rescales its actor's [-1,1] tanh output to the env's declared action
space. `config/ur10_gripper/moc_ur10_env_cfg.py`'s arm action `scale=0.0002` was tuned assuming
that upstream *100 rescale (see the comment there: Isaac Lab falls back to a [-100,100] Box for
this action term, since it declares no explicit bounds). `ACTION_SPACE_RESCALE` below reproduces
that same rescale so the physical per-step arm delta stays consistent with what's already tuned.
"""

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

ACTION_SPACE_RESCALE = 100.0


def _mlp(sizes, activation=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        net_arch=(256, 256, 256),
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,  # squashing (below) replaces skrl's hard-clamp
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction="sum",
        )

        self.trunk = _mlp([self.num_observations, *net_arch])
        self.mean_head = nn.Linear(net_arch[-1], self.num_actions)
        self.log_std_head = nn.Linear(net_arch[-1], self.num_actions)

    def compute(self, inputs, role):
        features = self.trunk(inputs["states"])
        return self.mean_head(features), self.log_std_head(features), {}

    def act(self, inputs, role=""):
        # raw_actions is the pre-squash sample from Normal(mean, exp(log_std)); log_prob is
        # computed under that same unsquashed distribution -- both needed for the tanh
        # correction below (correcting AFTER the fact, rather than duplicating the mixin's
        # distribution/sampling logic here).
        raw_actions, log_prob, outputs = GaussianMixin.act(self, inputs, role)

        squashed = torch.tanh(raw_actions)
        # SAC tanh-squash log-prob correction: log pi(a|s) = log mu(u|s) - sum(log(1 - tanh(u)^2))
        correction = torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = log_prob - correction

        outputs["mean_actions_squashed"] = torch.tanh(outputs["mean_actions"]) * ACTION_SPACE_RESCALE
        return squashed * ACTION_SPACE_RESCALE, log_prob, outputs


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, net_arch=(256, 256, 256)):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = _mlp([self.num_observations + self.num_actions, *net_arch, 1])

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


def build_models(observation_space, action_space, device, net_arch=(256, 256, 256)):
    """Builds the 5 models skrl's SAC agent expects: policy + 2 critics + 2 target critics.

    Target critics share the Critic class (same architecture) -- SAC.__init__ hard-copies
    critic_1/critic_2's parameters into them on construction (polyak=1), so their initial
    weights here don't matter.
    """
    return {
        "policy": Policy(observation_space, action_space, device, net_arch),
        "critic_1": Critic(observation_space, action_space, device, net_arch),
        "critic_2": Critic(observation_space, action_space, device, net_arch),
        "target_critic_1": Critic(observation_space, action_space, device, net_arch),
        "target_critic_2": Critic(observation_space, action_space, device, net_arch),
    }
