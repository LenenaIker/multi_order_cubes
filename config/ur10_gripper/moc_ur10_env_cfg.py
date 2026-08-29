from __future__ import annotations

import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp.actions import JointPositionToLimitsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG  # type: ignore
from isaaclab.markers.config import FRAME_MARKER_CFG


from ... import mdp
from ...moc_env_cfg import MOCEnvCfg, ObjectTableSceneCfg


def _assets_dir() -> str:
    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(pkg_dir, "assets")


def _asset(name: str) -> str:
    return os.path.join(_assets_dir(), name)



_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
    # Without this, a cube held under a constant (unchanging) grip loses kinetic
    # energy and PhysX puts it to sleep, which silently zeroes out its contact-force
    # reporting even though the physical contact is still real -- exactly the state a
    # grasp signal most needs to stay nonzero during. Confirmed live via
    # ContactSensorInspector.py (2026-08-29): force stopped decaying to 0 under a
    # held squeeze once this was set.
    sleep_threshold=0.0,
)


@configclass
class UR10LongSuctionMOCSceneCfg(ObjectTableSceneCfg):
    cube_light_s = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_light_s",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.00, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_asset("blue_block_light.usd"), scale=mdp.SCALES["s"], rigid_props=_RIGID_PROPS),
    )
    cube_light_m = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_light_m",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.00, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_asset("blue_block_light.usd"), scale=mdp.SCALES["m"], rigid_props=_RIGID_PROPS),
    )
    cube_light_l = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_light_l",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.00, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_asset("blue_block_light.usd"), scale=mdp.SCALES["l"], rigid_props=_RIGID_PROPS),
    )

    cube_flat_s = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_flat_s",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_asset("blue_block_flat.usd"), scale=mdp.SCALES["s"], rigid_props=_RIGID_PROPS),
    )
    cube_flat_m = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_flat_m",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_asset("blue_block_flat.usd"), scale=mdp.SCALES["m"], rigid_props=_RIGID_PROPS),
    )
    cube_flat_l = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_flat_l",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_asset("blue_block_flat.usd"), scale=mdp.SCALES["l"], rigid_props=_RIGID_PROPS),
    )

    cube_dark_s = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_dark_s",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_asset("blue_block_dark.usd"), scale=mdp.SCALES["s"], rigid_props=_RIGID_PROPS),
    )
    cube_dark_m = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_dark_m",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_asset("blue_block_dark.usd"), scale=mdp.SCALES["m"], rigid_props=_RIGID_PROPS),
    )
    cube_dark_l = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_dark_l",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_asset("blue_block_dark.usd"), scale=mdp.SCALES["l"], rigid_props=_RIGID_PROPS),
    )


@configclass
class UR10LongSuctionMultiOrderCubesEnvCfg(MOCEnvCfg):
    scene: UR10LongSuctionMOCSceneCfg = UR10LongSuctionMOCSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=False,
    )

    def __post_init__(self):
        super().__post_init__()

        x0, y0, _ = self.slot_positions[0]
        parked_pos = [x0, y0, -10.0]

        for name in mdp.CUBE_KEYS_9:
            getattr(self.scene, name).init_state.pos = parked_pos

        self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.left_finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/left_inner_finger",
            force_threshold=1.0,
            history_length=self.decimation,
        )
        self.scene.right_finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/right_inner_finger",
            force_threshold=1.0,
            history_length=self.decimation,
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/TcpFrame"
        marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/robotiq_base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg, # Eje pequeño
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/robotiq_base_link",
                    name="tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.18)),
                ),
            ],
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="robotiq_base_link",
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.18),
            ),
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            # SB3's SAC actor outputs a tanh-bounded [-1,1] value, which gets linearly rescaled to
            # the env's action space (Isaac Lab falls back to a fixed [-100,100] Box for any
            # unbounded action term, see Sb3VecEnvWrapper.__init__). process_actions multiplies
            # that already-rescaled raw action directly by `scale` with no further clamp, so the
            # true max per-step delta is raw_action_bound * scale, not scale itself. Divided by
            # ~100 here so the actor's full natural output range maps to the intended physical
            # delta (2cm / ~2.9deg per step) instead of requiring the policy to live in a ~1%
            # sliver near zero to avoid physically absurd multi-meter jumps.
            scale=(0.0002, 0.0002, 0.0002, 0.0005, 0.0005, 0.0005),
            debug_vis=False,
        )
        
        self.actions.gripper = JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            rescale_to_limits=True,
        )

        self.actions.next_action = mdp.NextFlagActionCfg(asset_name="robot")
        self.sim.render_interval = 5