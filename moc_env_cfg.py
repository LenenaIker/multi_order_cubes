from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

import torch


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and objects."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # arm_action: mdp.JointPositionActionCfg = MISSING
    # gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

    # NEW: scalar action used as NEXT "commit" signal
    # If you implement NextFlagActionCfg (see below), you can add it here.
    # If not, you can omit this and set env.moc_next_signal yourself in your env wrapper.
    # next_action: mdp.NextFlagActionCfg = MISSING
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        obs = ObsTerm(func=mdp.policy_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass
    # time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # IMPORTANT: remove success termination (no auto-termination on success)
    # success = DoneTerm(func=mdp.move_success)


@configclass
class RewardsCfg:
    # Existing dense shaping
    # ee_to_cube = RewTerm(func=mdp.reward_ee_to_cube, weight=1.0)
    # cube_to_slot = RewTerm(func=mdp.reward_target_cube_to_to_slot, weight=1.5)
    # action_l2 = RewTerm(func=mdp.reward_penalty_action_l2, weight=1.0)

    # IMPORTANT: remove/disable old "success bonus" if it was state-based.
    # success = RewTerm(func=mdp.reward_success_bonus, weight=1.0)

    # NEW: NEXT / commit rewards
    next_commit_success = RewTerm(
        func=mdp.reward_next_commit_success,
        weight=1.0,
        params=dict(
            tau=0.0,                 # threshold on moc_next_signal if in [-1,1]
            stable_window=3,         # M steps stable
            cooldown_steps=8,        # anti-spam
            R_commit=8.0,            # positive reward for correct NEXT
            advance_command=True,    # resample command on correct commit
        ),
    )

    next_commit_fail = RewTerm(
        func=mdp.reward_next_commit_fail,
        weight=1.0,
        params=dict(
            tau=0.0,
            stable_window=3,
            cooldown_steps=8,
            R_false=2.0,             # penalty for premature NEXT
        ),
    )

    wait_after_success = RewTerm(
        func=mdp.reward_wait_after_success,
        weight=1.0,
        params=dict(
            stable_window=3,
            lambda_wait=0.05,        # small negative while success is stable but NEXT not fired
        ),
    )
    # Penalize disturbing non-target cubes (reduces pushing/bulldozing behavior)
    disturb_other_cubes = RewTerm(
        func=mdp.reward_penalty_disturb_other_cubes,
        weight=1.0,
        params=dict(
            lambda_disturb=0.25,
            tol_xy=0.01,
        ),
    )

@configclass
class EventsCfg:
    """Events (reset-time hooks)."""

    randomize_cubes = EventTerm(
        func=mdp.randomize_cubes_on_slots,
        mode="reset",
    )

    sample_from_to = EventTerm(
        func=mdp.sample_from_to_on_reset,
        mode="reset",
        params={},  # no params needed
    )



@configclass
class MOCEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the multi-order cubes environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # commands: CommandCfg = CommandCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()

    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        self.slot_positions = torch.tensor(
            [
                [0.6, 0.3, 0.021],
                [0.6, 0.1, 0.021],
                [0.6, -0.1, 0.021],
                [0.6, -0.3, 0.021],
            ],
            dtype=torch.float32,
        )

        self.decimation = 5
        self.episode_length_s = 30.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
