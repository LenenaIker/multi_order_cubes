from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTermCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

_NEXT_THRESHOLD = 0.5
_NEXT_COOLDOWN_STEPS = 30


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    left_finger_contact: ContactSensorCfg = MISSING
    right_finger_contact: ContactSensorCfg = MISSING

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


@configclass
class ActionsCfg:
    arm_action: ActionTermCfg = MISSING
    gripper: JointPositionActionCfg = MISSING
    next_action: mdp.NextFlagActionCfg = MISSING


@configclass
class ObservationsCfg:
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
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_off_table = DoneTerm(
        func=mdp.cube_fell_off_table,
        time_out=False,
        params=dict(height_threshold=-0.05),
    )


# Reward Weights
WEIGHT_REACH_XY = 12.0
WEIGHT_REACH_Z = 9.5
WEIGHT_LIFT = 15.0
WEIGHT_GRASP = 10.0
WEIGHT_NEXT = 1.0

@configclass
class RewardsCfg:
    reach_xy_abs = RewTerm(
        func=mdp.reward_reach_xy_rational,
        weight=WEIGHT_REACH_XY,
        params=dict(k_xy=0.12, p=1.0, weight=WEIGHT_REACH_XY),
    )

    reach_z = RewTerm(
        func=mdp.reward_reach_z_gated,
        weight=WEIGHT_REACH_Z,
        params=dict(k_z=0.06, p=1.0, gate_dxy=0.18, gate_band=0.05, flat_margin=0.03, weight=WEIGHT_REACH_Z),
    )


    object_lifted = RewTerm(
        func=mdp.reward_object_lifted,
        weight=WEIGHT_LIFT,
        params=dict(target_height=0.15, tolerance=0.0001, weight=WEIGHT_LIFT),
    )

    grasp_contact = RewTerm(
        func=mdp.reward_grasp_contact,
        weight=WEIGHT_GRASP,
        params=dict(success_xy=0.2, success_z=0.2, force_cap=20.0, weight=WEIGHT_GRASP),
    )

    next_signal = RewTerm(
        func=mdp.reward_next_signal,
        weight=WEIGHT_NEXT,
        params=dict(
            next_threshold=_NEXT_THRESHOLD,
            cooldown_steps=_NEXT_COOLDOWN_STEPS,
            success_xy=0.05,
            success_z=0.03,
            bonus=5.0,
            penalty=-0.03,
            weight=WEIGHT_NEXT,
        ),
    )

    grip_distance_diag = RewTerm(
        # Isaac Lab's RewardManager skips any term whose weight is exactly 0.0 (a micro-
        # optimization) without ever calling its function, so a real "does nothing" weight
        # can't be 0.0. Any nonzero value is harmless here regardless: the function always
        # returns torch.zeros(...), so weight * 0 == 0 no matter what weight is.
        func=mdp.diag_grip_distance,
        weight=1.0,
        params=dict(closed_threshold=0.7),
    )

    bystander_displacement = RewTerm(
        func=mdp.penalty_bystander_displacement,
        weight=1.0,
        params=dict(tolerance_xy=0.02, weight_per_m=5.0),
    )

    # table_proximity = RewTerm(
    #     func=mdp.penalty_table_proximity,
    #     weight=1.0,
    #     params=dict(safe_height=0.015, weight_per_m=20.0, max_excess=0.03),
    # )

    arm_joint_vel_penalty = RewTerm(
        func=mdp.penalty_arm_joint_velocity,
        weight=-0.005,
        params=dict(
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            max_l2=10.0,
        ),
    )

@configclass
class EventsCfg:
    moc_reset = EventTerm(func=mdp.moc_reset_on_reset, mode="reset", params={})

    moc_next_flag = EventTerm(
        func=mdp.consume_next_signal,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # placeholder; set to (step_dt, step_dt) in __post_init__
        is_global_time=False,
        params=dict(next_threshold=_NEXT_THRESHOLD, cooldown_steps=_NEXT_COOLDOWN_STEPS),
    )

    moc_activate_finger_contacts = EventTerm(
        func=mdp.activate_finger_contact_sensors,
        mode="prestartup",
        params={},
    )


@configclass
class MOCEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
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
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2

        step_dt = self.decimation * self.sim.dt
        self.events.moc_next_flag.interval_range_s = (step_dt, step_dt)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625