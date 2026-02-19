from __future__ import annotations

import os

from isaaclab.assets import RigidObjectCfg, SurfaceGripperCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import SurfaceGripperBinaryActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.universal_robots import UR10_LONG_SUCTION_CFG  # type: ignore

from ...moc_env_cfg import MOCEnvCfg, ObjectTableSceneCfg
from ... import mdp


def _assets_dir() -> str:
    """
    Returns absolute path to: multi_order_cubes/assets
    This file lives in: multi_order_cubes/config/ur10_gripper/
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # .../multi_order_cubes
    return os.path.join(pkg_dir, "assets")


def _asset(name: str) -> str:
    """Absolute path to an asset in multi_order_cubes/assets."""
    return os.path.join(_assets_dir(), name)


@configclass
class UR10LongSuctionMOCSceneCfg(ObjectTableSceneCfg):
    """Scene: Table + UR10 Long Suction + 3 cubes (fixed colors via local USDs)."""

    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.00, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            # LOCAL asset (light blue)
            usd_path=_asset("blue_block_light.usd"),
            scale=(0.80, 0.80, 0.80),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            # LOCAL asset (base/flat blue)
            usd_path=_asset("blue_block_flat.usd"),
            scale=(1.00, 1.00, 1.00),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            # LOCAL asset (dark blue)
            usd_path=_asset("blue_block_dark.usd"),
            scale=(1.20, 1.20, 1.20),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    surface_gripper: SurfaceGripperCfg = SurfaceGripperCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/SurfaceGripper",
        max_grip_distance=0.0075,
        shear_force_limit=5000.0,
        coaxial_force_limit=5000.0,
        retry_interval=0.05,
    )


@configclass
class UR10LongSuctionMultiOrderCubesEnvCfg(MOCEnvCfg):
    """
    UR10 LONG + Suction, joint position control, with NEXT action enabled.
    One single Gym environment entry-point.
    """

    scene: UR10LongSuctionMOCSceneCfg = UR10LongSuctionMOCSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=False,
    )

    def __post_init__(self):
        super().__post_init__()

        # Slot placement (your existing logic)
        self.scene.cube_1.init_state.pos = self.slot_positions[0]
        self.scene.cube_2.init_state.pos = self.slot_positions[1]
        self.scene.cube_3.init_state.pos = self.slot_positions[2]

        # Suction grippers currently require CPU simulation
        self.device = "cpu"

        # Robot
        self.scene.robot = UR10_LONG_SUCTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # End-effector frame transformer
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.22, 0.0, 0.0]),
                ),
            ],
        )

        # Actions
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )

        # NEXT action
        self.actions.next_action = mdp.NextFlagActionCfg(asset_name="robot")

        # Render interval (performance)
        self.sim.render_interval = 5
