import argparse
from isaaclab.app import AppLauncher

# -------------------------------------------------
# CLI
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -------------------------------------------------
# Launch app FIRST
# -------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------
# Imports AFTER app creation
# -------------------------------------------------
import torch  # noqa: E402
import omni.ui as ui  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from multi_order_cubes.config.ur10_gripper import UR10LongSuctionMultiOrderCubesEnvCfg  # noqa: E402
from multi_order_cubes.mdp.commands import latch_target_cube_from_command  # noqa: E402
from multi_order_cubes.mdp.step_cache import invalidate_moc_cache
from multi_order_cubes import mdp  # noqa: E402



# =================================================
# Small debug HUD
# =================================================
class RewardDebugHUD:
    def __init__(self):
        self.window = ui.Window(
            "MOC Debug Overlay",
            width=360,
            height=320,
            position_x=20,
            position_y=20,
        )

        self.labels = {}

        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Multi Order Cubes - Debug", height=24)

                self.labels["cmd"] = ui.Label("cmd: -", height=20)
                self.labels["target"] = ui.Label("target_cube_id: -", height=20)
                self.labels["tcp"] = ui.Label("tcp: -", height=20)
                self.labels["cube"] = ui.Label("cube: -", height=20)

                ui.Spacer(height=6)

                self.labels["dxy"] = ui.Label("d_xy: -", height=20)
                self.labels["adz"] = ui.Label("|d_z|: -", height=20)
                self.labels["gate"] = ui.Label("gate_xy: -", height=20)

                ui.Spacer(height=6)

                self.labels["rxy"] = ui.Label("r_xy: -", height=20)
                self.labels["rz"] = ui.Label("r_z: -", height=20)
                self.labels["rtotal"] = ui.Label("reward_total: -", height=20)

                ui.Spacer(height=6)

                self.labels["obs_shape"] = ui.Label("obs shape: -", height=20)

    def set_text(self, key: str, value: str):
        if key in self.labels:
            self.labels[key].text = value


def fmt_vec3(x: torch.Tensor) -> str:
    v = x[0].detach().cpu().tolist()
    return f"[{v[0]: .3f}, {v[1]: .3f}, {v[2]: .3f}]"


def main():
    env_cfg = UR10LongSuctionMultiOrderCubesEnvCfg()
    env_cfg.scene.num_envs = 1

    # -------------------------------------------------
    # IMPORTANT: CPU physics so Physics Inspector can drive joints
    # -------------------------------------------------
    env_cfg.sim.device = "cpu"
    env_cfg.sim.physx.use_gpu = False

    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs, info = env.reset()

    # -------------------------------------------------
    # Fixed command for debugging
    # -------------------------------------------------
    env.command_from_to = torch.tensor([[1, 2]], dtype=torch.long, device=env.device)
    latch_target_cube_from_command(env)

    hud = RewardDebugHUD()

    physics_dt = env.cfg.sim.dt
    i = 0

    while simulation_app.is_running():
        # No env.step(actions)
        env.sim.step()
        env.scene.update(physics_dt)

        invalidate_moc_cache(env)

        # IMPORTANTE: sin env.step(), el step_cache no se invalida solo
        if hasattr(env, "_moc_cache"):
            env._moc_cache.clear()
        if hasattr(env, "_moc_cache_token"):
            env._moc_cache_token = -1

        # ---------------------------------------------
        # Recompute observations manually
        # ---------------------------------------------
        obs = env.observation_manager.compute()

        # ---------------------------------------------
        # Recompute reward terms manually
        # Current cfg:
        # reach_xy_abs weight = 3.0
        # reach_z      weight = 2.0
        # ---------------------------------------------
        r_xy = mdp.reward_reach_xy_rational(env, k_xy=0.25, p=1.0)
        
        r_z = mdp.reward_reach_z_gated(
            env,
            sigma_z=0.08,
            gate_dxy=0.16,
            gate_band=0.06,
        )

        r_total = 6.0 * r_xy + 6.0 * r_z

        # ---------------------------------------------
        # Read extra debug values saved by rewards.py
        # ---------------------------------------------
        dxy = env.extras.get("moc/reach_dist_xy", None) if hasattr(env, "extras") and env.extras else None
        gate = env.extras.get("moc/reach_gate_xy", None) if hasattr(env, "extras") and env.extras else None
        adz = env.extras.get("moc/reach_abs_dz", None) if hasattr(env, "extras") and env.extras else None

        # ---------------------------------------------
        # Read TCP and target cube positions
        # ---------------------------------------------
        tcp = mdp.get_tcp_pos_w(env, ee_frame_name="ee_frame")
        active_cubes = mdp.get_active_cube_pos_w(env)

        target_id = env.target_cube_id.to(torch.long).clamp(0, active_cubes.shape[1] - 1)
        row = torch.arange(env.num_envs, device=env.device)
        target_cube = active_cubes[row, target_id, :]

        # ---------------------------------------------
        # Command text
        # ---------------------------------------------
        cmd = env.command_from_to[0].detach().cpu().tolist()
        target_cube_id = int(env.target_cube_id[0].item())

        # ---------------------------------------------
        # Obs shape text
        # ---------------------------------------------
        if isinstance(obs, dict):
            obs_desc = []
            for k, v in obs.items():
                if isinstance(v, dict):
                    inner = ", ".join([f"{ik}:{tuple(iv.shape)}" for ik, iv in v.items()])
                    obs_desc.append(f"{k}{{{inner}}}")
                else:
                    obs_desc.append(f"{k}:{tuple(v.shape)}")
            obs_shape_str = " | ".join(obs_desc)
        else:
            obs_shape_str = str(tuple(obs.shape))

        # ---------------------------------------------
        # Update HUD
        # ---------------------------------------------
        hud.set_text("cmd", f"cmd: {cmd[0]} -> {cmd[1]}")
        hud.set_text("target", f"target_cube_id: {target_cube_id}")
        hud.set_text("tcp", f"tcp: {fmt_vec3(tcp)}")
        hud.set_text("cube", f"target_cube: {fmt_vec3(target_cube)}")

        if dxy is not None:
            hud.set_text("dxy", f"d_xy: {dxy[0].item():.5f}")
        if adz is not None:
            hud.set_text("adz", f"|d_z|: {adz[0].item():.5f}")
        if gate is not None:
            hud.set_text("gate", f"gate_xy: {gate[0].item():.5f}")

        hud.set_text("rxy", f"r_xy: {r_xy[0].item():.5f}")
        hud.set_text("rz", f"r_z: {r_z[0].item():.5f}")
        hud.set_text("rtotal", f"reward_total: {r_total[0].item():.5f}")
        hud.set_text("obs_shape", f"obs shape: {obs_shape_str}")

        # ---------------------------------------------
        # Optional terminal print every 20 frames
        # ---------------------------------------------
        if i % 20 == 0:
            print(
                f"[{i:06d}] "
                f"cmd={cmd[0]}->{cmd[1]}  "
                f"target={target_cube_id}  "
                f"dxy={(dxy[0].item() if dxy is not None else float('nan')):.5f}  "
                f"|dz|={(adz[0].item() if adz is not None else float('nan')):.5f}  "
                f"gate={(gate[0].item() if gate is not None else float('nan')):.5f}  "
                f"r_xy={r_xy[0].item():.5f}  "
                f"r_z={r_z[0].item():.5f}  "
                f"total={r_total[0].item():.5f}"
            )

        i += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# 9.2
# -54.1
# 111.4
# -46.2
# 96.3
# 0.0