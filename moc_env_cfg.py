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

    policy: PolicyCfg = PolicyCfg()



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_fell_off_table = DoneTerm(
        func=mdp.cube_fell_off_table,
        params=dict(
            z_margin_below_slots=0.15,
            xy_margin=0.35,
        ),
    )

    ee_below_table = DoneTerm(
        func=mdp.ee_below_table,
        params=dict(
            table_z=0.0199,
            z_margin_below_slots=0.002
        )
    )



@configclass
class RewardsCfg:
    # Dense shaping: move EE towards the target cube
    ee_to_target_xy = RewTerm(
        func=mdp.reward_shaping_ee_to_target_xy,
        weight=1.0,
        params=dict(
            sigma_xy=0.15,
            scale=1.0,
        ),
    )
    
    # Dense shaping 3D: EE towards a pregrasp point above target cube
    ee_to_target_pregrasp = RewTerm(
        func=mdp.reward_shaping_ee_to_target_pregrasp,
        weight=0.5,  # pequeño, no debe dominar
        params=dict(
            sigma=0.20,
            scale=1.0,
            z_offset=0.08,
        ),
    )

    # Dense shaping: move target cube towards the commanded to_slot
    target_to_to_slot_xy = RewTerm(
        func=mdp.reward_shaping_target_to_to_slot_xy_gated_by_suction,
        weight=2.0,
        params=dict(sigma=0.20, scale=1.0),
    )

    # Penalize disturbing non-target cubes (reduces pushing/bulldozing behavior)
    # Keep this BEFORE next_commit_success so its baseline uses the *current* command.
    disturb_other_cubes = RewTerm(
        func=mdp.reward_penalty_disturb_other_cubes,
        weight=0.1,
        params=dict(
            lambda_disturb=0.25,
            tol_xy=0.01,
        ),
    )

    # far_from_target_penalty = RewTerm(
    #     func=mdp.reward_penalty_far_from_target,
    #     weight=1.0,
    #     params=dict(
    #         sigma=0.2,
    #         lambda_far=0.5,
    #     ),
    # )

    dist_xy_l2_penalty = RewTerm(
        func=mdp.reward_penalty_dist_xy_l2,
        weight=1.0,
        params=dict(
            sigma_xy=0.25,
            scale=1.0,
        ),
    )

    # 3D pregrasp shaping (tip-centric)
    ee_to_target_pregrasp_3d = RewTerm(
        func=mdp.reward_shaping_ee_to_target_pregrasp_3d,
        weight=1.0,
        params=dict(
            sigma=0.20,
            scale=1.0,
            z_offset=0.08,
        ),
    )

    # Reward using suction when close to target (tool usage)
    suction_near_target = RewTerm(
        func=mdp.reward_suction_near_target,
        weight=1,
        params=dict(
            sigma=0.08,
            # Reduce "hover" reward
            scale_proximity=0.5,
            # Make "try suction correctly" very attractive
            scale_bonus_if_suction_cmd=4.0,
            scale_bonus_if_suction_on=4.0,
            # Alignment shaping (X+ should point to world up)
            scale_align=2.0,
            align_power=4.0,
        )
    )
        
    # Reward lifting target once suction is active
    lift_target_when_suction = RewTerm(
        func=mdp.reward_lift_target_when_suction,
        weight=4.0,
        params=dict(
            z_lift_min=0.03,
            z_lift_max=0.15,
            scale=1.0,
        ),
    )

    next_by_phase = RewTerm(
        func=mdp.reward_next_by_phase,
        weight=1.0,
        params=dict(
            tau=0.0,
            cooldown_steps=30,
            R_next_ok=8.0,
            R_next_bad=3.0,
            advance_command=True,
        ),
    )

    joint_vel_penalty = RewTerm(
        func=mdp.reward_penalty_joint_vel,
        weight=0.07,
        params=dict(lambda_vel=0.01),
    )

    # Atracción fuerte y suave (reach-style)
    reach_xy = RewTerm(
        func=mdp.reward_reach_xy_tanh,
        weight=2.0,
        params=dict(std_xy=0.25),
    )

    reach_3d = RewTerm(
        func=mdp.reward_reach_3d_tanh,
        weight=1.0,
        params=dict(std_3d=0.35),
    )

    # Empuja a BAJAR a la altura de grasp (clave para llegar a contacto)
    grasp_height = RewTerm(
        func=mdp.reward_grasp_height_tanh,
        weight=2.0,
        params=dict(grasp_z_offset=0.03, std_z=0.06),
    )

    # Succión solo cuando estás en zona de grasp (y penaliza fuera)
    suction_cmd_gated = RewTerm(
        func=mdp.reward_suction_cmd_gated,
        weight=3.0,
        params=dict(gain=1.0, xy_tol=0.04, z_tol=0.05, grasp_z_offset=0.03),
    )

    suction_cmd_outside_penalty = RewTerm(
        func=mdp.penalty_suction_cmd_outside,
        weight=1.0,
        params=dict(penalty=0.2, xy_tol=0.06, z_tol=0.07, grasp_z_offset=0.03),
    )

    moc_metrics_logger = RewTerm(
        func=mdp.reward_log_metrics,
        weight=0.0,
        params={},
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
