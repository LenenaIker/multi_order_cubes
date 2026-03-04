from __future__ import annotations

import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointPositionToLimitsActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG  # type: ignore

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

# -------------------------
# helper constants
# -------------------------
_SCALES = {
    "s": (0.80, 0.80, 0.80),
    "m": (1.00, 1.00, 1.00),
    "l": (1.20, 1.20, 1.20),
}

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
    """Scene: Table + UR10 Long Suction + 9 cubes (3 colors x 3 sizes)."""

    # -------------------------
    # 9 cubes: light/flat/dark x s/m/l
    # -------------------------
    cube_light_s = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_light_s",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.00, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_asset("blue_block_light.usd"),
            scale=_SCALES["s"],
            rigid_props=_RIGID_PROPS,
        ),
    )
    cube_light_m = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_light_m",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.00, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_asset("blue_block_light.usd"),
            scale=_SCALES["m"],
            rigid_props=_RIGID_PROPS,
        ),
    )
    cube_light_l = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_light_l",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.00, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_asset("blue_block_light.usd"),
            scale=_SCALES["l"],
            rigid_props=_RIGID_PROPS,
        ),
    )

    cube_flat_s = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_flat_s",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_asset("blue_block_flat.usd"),
            scale=_SCALES["s"],
            rigid_props=_RIGID_PROPS,
        ),
    )
    cube_flat_m = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_flat_m",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_asset("blue_block_flat.usd"),
            scale=_SCALES["m"],
            rigid_props=_RIGID_PROPS,
        ),
    )
    cube_flat_l = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_flat_l",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_asset("blue_block_flat.usd"),
            scale=_SCALES["l"],
            rigid_props=_RIGID_PROPS,
        ),
    )

    cube_dark_s = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_dark_s",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_asset("blue_block_dark.usd"),
            scale=_SCALES["s"],
            rigid_props=_RIGID_PROPS,
        ),
    )
    cube_dark_m = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_dark_m",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_asset("blue_block_dark.usd"),
            scale=_SCALES["m"],
            rigid_props=_RIGID_PROPS,
        ),
    )
    cube_dark_l = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_dark_l",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.03], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_asset("blue_block_dark.usd"),
            scale=_SCALES["l"],
            rigid_props=_RIGID_PROPS,
        ),
    )


    

@configclass
class UR10LongSuctionMultiOrderCubesEnvCfg(MOCEnvCfg):
    """
    UR10 LONG + Suction, joint position control, with NEXT action enabled.
    One single Gym environment entry-point.
    """

    scene: UR10LongSuctionMOCSceneCfg = UR10LongSuctionMOCSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=False,
    )

    def __post_init__(self):
        super().__post_init__()

        # Park ALL cube instances at startup; the reset event will activate 3 of them.
        cube_names = [
            "cube_light_s", "cube_light_m", "cube_light_l",
            "cube_flat_s",  "cube_flat_m",  "cube_flat_l",
            "cube_dark_s",  "cube_dark_m",  "cube_dark_l",
        ]

        x0, y0, _ = self.slot_positions[0]
        parked_pos = [x0, y0, -10.0]  # far below the table

        for name in cube_names:
            getattr(self.scene, name).init_state.pos = parked_pos


        # Robot
        self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---- FORCE ARM ACTUATOR LIMITS (UR10e sometimes has no explicit effort/vel limits in cfg)
        robot_cfg = self.scene.robot

        # Use conservative but non-zero limits; you can tune later.
        # Values aligned with typical UR10e joint drive maxForce seen in USD (~330) and reasonable vel.
        for key in ["shoulder", "elbow", "wrist"]:
            if key in robot_cfg.actuators:
                act = robot_cfg.actuators[key]
                # Set only if missing/zero to avoid overriding if already correct in your build
                if getattr(act, "effort_limit_sim", None) in (None, 0.0):
                    act.effort_limit_sim = 330.0
                if getattr(act, "velocity_limit_sim", None) in (None, 0.0):
                    act.velocity_limit_sim = 3.0

        # End-effector frame transformer (TIP via offset from ee_link rigid body)
        self.scene.ee_frame = FrameTransformerCfg(
            # anchor on a rigid body
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/robotiq_base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/robotiq_base_link",
                    name="tcp",
                    # offset guess: forward/down from base_link to the pinch midpoint
                    # YOU MUST tune these numbers once with debug_vis
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.12)),
                ),
            ]
        )
        # Actions
        self.actions.arm_action = JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            # acción en [-1,1] -> límites del joint; opcionalmente reduce sensibilidad
            # scale={".*": 1.0},
            # clip={".*": (-1.0, 1.0)},
            rescale_to_limits=True,
        )

        self.actions.gripper = JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            # scale={".*": 1.0},
            # clip={".*": (-1.0, 1.0)},
            rescale_to_limits=True,
        )
        
        # Robotiq: controla/observa el dedo principal
        # gripper_joint_names = ["finger_joint"]

        # NEXT action
        self.actions.next_action = mdp.NextFlagActionCfg(asset_name="robot")

        # Render interval (performance)
        self.sim.render_interval = 5
