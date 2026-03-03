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
from isaaclab.envs.mdp.actions import JointPositionActionCfg

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
    arm_action: JointPositionActionCfg = MISSING
    gripper: JointPositionActionCfg = MISSING
    next_action: mdp.NextFlagActionCfg = MISSING


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

    policy: PolicyCfg = PolicyCfg()



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # cube_fell_off_table = DoneTerm(
    #     func=mdp.cube_fell_off_table,
    #     params=dict(
    #         z_margin_below_slots=0.15,
    #         xy_margin=0.35,
    #     ),
    # )


@configclass
class RewardsCfg:
    tip_to_target_xy = RewTerm(
        func=mdp.reward_tip_to_target_xy,
        weight=15.0,
        params=dict(sigma_xy=0.15),
    )

    tip_to_target_z = RewTerm(
        func=mdp.reward_tip_to_target_z,
        weight=6.0,
        params=dict(z_offset=0.10, sigma_z=0.06),
    )

    close_cmd_in_zone = RewTerm(
        func=mdp.reward_close_cmd_in_grasp_zone,
        weight=0.0,#2.0,
        params=dict(radius_xy=0.04, z_lo=-0.01, z_hi=0.03),
    )

    close_cmd_outside_penalty = RewTerm(
        func=mdp.penalty_close_cmd_outside_grasp_zone,
        weight=0.0,#1.0,
        params=dict(radius_xy=0.06, z_lo=-0.02, z_hi=0.05),
    )

    # 2) Lift: Z del cubo (gated por suction_on)
    lift_target = RewTerm(
        func=mdp.reward_lift_target_z,
        weight=0.0,#10.0,
        params=dict(
            z_lift_min=0.02,
            z_lift_max=0.12,
        ),
    )

    # 3) Transport: cubo -> slot destino (XY), gated por suction_on + lifted
    move_target_to_goal_xy = RewTerm(
        func=mdp.reward_move_target_to_goal_xy,
        weight=0.0,#12.0,
        params=dict(
            sigma_xy=0.12,
            z_lift_gate=0.02,
        ),
    )

    # 4) NEXT: commit cuando la tarea está establemente completada
    next_commit = RewTerm(
        func=mdp.reward_next_commit,
        weight=0.0,#1.0,
        params=dict(
            tau=0.0,
            stable_window=6,
            cooldown_steps=40,
            
            R_next_ok=15.0,
            R_next_bad=0.2,
            R_wait=0.01,

            tol_xy=0.03,
            tol_z=0.05,
            require_to_clear=True,
            require_settled=True,
            vel_tol=0.25,
        ),
    )

    # Penalización suave para estabilidad
    joint_vel_penalty = RewTerm(
        func=mdp.reward_penalty_joint_vel,
        weight=0.002,
        params=dict(lambda_vel=0.003),
    )



@configclass
class EventsCfg:
    """Events (reset-time hooks)."""

    moc_reset = EventTerm(
        func=mdp.moc_reset_on_reset,
        mode="reset",
        params={},
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
                [0.87, 0.3, 0.021],
                [0.87, 0.1, 0.021],
                [0.87, -0.1, 0.021],
                [0.87, -0.3, 0.021],
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
