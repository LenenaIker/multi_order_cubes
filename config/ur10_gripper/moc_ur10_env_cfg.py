from __future__ import annotations

import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp.actions import JointPositionToLimitsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab.sensors import FrameTransformerCfg
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
            scale=(0.02, 0.02, 0.02, 0.05, 0.05, 0.05),
            debug_vis=False,
        )
        
        self.actions.gripper = JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            rescale_to_limits=True,
        )

        self.actions.next_action = mdp.NextFlagActionCfg(asset_name="robot")
        self.sim.render_interval = 5