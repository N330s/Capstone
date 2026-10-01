"""
Manual OpenArm tabletop pick-and-insert demonstration collector using Cartesian XYZ + gripper control.

Task setup:
  - Plug starts FREE on the tabletop, not in the robot gripper.
  - Socket is placed by the same deterministic collection scene bank as collect_data.py.
  - The operator performs the complete task: approach -> grasp -> lift -> transport -> insert.

Keyboard:
  W/S = +X/-X       A/D = +Y/-Y       R/F = +Z/-Z
  Q/E = yaw -/+     O/C = open/close gripper
  TAB = fine/coarse  SPACE = pause
  N = reset scene   ENTER = save after SUCCESS
  X = abort         ESC = quit

The recorded dataset action remains:
  [7 right-arm joint targets (rad), finger travel (m)]
because that is OpenArmInsertEnv.step()'s action interface. The human controls
Cartesian XYZ; OpenArmInsertEnv.solve_ik() converts XYZ+orientation to joints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


import pygame
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.openarm_insert import OpenArmInsertEnv
from data_pipeline.scene_bank import as_reset_options, collection_bank, workspace_for_env
from data_pipeline.episodes import save_episode


DEFAULT_OUTPUT = Path("data/manual_demos")
CONTROL_HZ = 30.0
DEFAULT_EPISODES = 10
DEFAULT_START = 0
DEFAULT_EPISODE_LIMIT_S = 120.0

# Cartesian movement speed.
# The multiplier is applied to both fine/coarse translational and yaw speeds.
DEFAULT_SPEED_MULTIPLIER = 2.0

FINE_SPEED_M_S = 0.015
COARSE_SPEED_M_S = 0.060
FINE_YAW_DEG_S = 12.0
COARSE_YAW_DEG_S = 45.0


class CartesianCollector:
    def __init__(
        self,
        output: Path,
        episodes: int,
        start: int,
        episode_limit_s: float,
        fine_speed: float,
        coarse_speed: float,
        fine_yaw: float,
        coarse_yaw: float,
        speed_multiplier: float,
    ):
        self.output = Path(output)
        self.n_episodes = episodes
        self.start_index = start
        self.episode_limit_s = episode_limit_s

        self.fine_speed = fine_speed
        self.coarse_speed = coarse_speed
        self.fine_yaw = fine_yaw
        self.coarse_yaw = coarse_yaw
        self.speed_multiplier = max(0.05, float(speed_multiplier))

        self.env = OpenArmInsertEnv(images=True)

        # Runtime-only override: no need to edit openarm_v1.json.
        self.env.config["episode_limit_s"] = self.episode_limit_s

        # Use the EXACT same collection-bank construction as collect_data.py.
        # This guarantees the Cartesian collector sees the same tabletop scenes,
        # seed numbering, plug placement, socket placement, and visual randomization.
        self.ws = workspace_for_env(self.env)
        self.scenes = collection_bank(
            self.start_index + self.n_episodes,
            ws=self.ws,
            start=0,
        )[self.start_index:self.start_index + self.n_episodes]

        self.viewer = None
        self.running = True
        self.paused = False
        self.coarse = False

        self.target_xyz = None
        self.target_rot = None
        self.target_finger = None

        self.screen = None
        self.font = None

    def init_ui(self):
        pygame.init()
        pygame.display.set_caption("OpenArm Cartesian Demonstration Collector")
        self.screen = pygame.display.set_mode((820, 300))
        self.font = pygame.font.SysFont("consolas", 18)

    def draw_status(self, episode, scene_index, info):
        if self.screen is None:
            return

        self.screen.fill((25, 25, 25))

        speed = (self.coarse_speed if self.coarse else self.fine_speed) * self.speed_multiplier
        yaw = (self.coarse_yaw if self.coarse else self.fine_yaw) * self.speed_multiplier

        lines = [
            f"Episode {episode} | Scene {scene_index}",
            f"XYZ speed: {speed:.3f} m/s | Yaw: {yaw:.1f} deg/s | "
            f"Mode: {'COARSE' if self.coarse else 'FINE'}",
            "",
            "W/S = X+/-     A/D = Y-/+     R/F = Z-/+",
            "Q/E = Yaw-/+   O = Open      C = Close",
            "TAB = Fine/Coarse    SPACE = Pause    N = Reset",
            "ENTER = Save after SUCCESS    X = Abort    ESC = Quit",
            "",
        ]

        if self.target_xyz is not None:
            p = self.target_xyz
            lines.append(f"Target XYZ: [{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}]")

        if self.target_finger is not None:
            lines.append(f"Finger travel: {self.target_finger:.4f} m")

        lines.append(
            f"Outcome: {info.get('outcome', 'running')} | "
            f"Grasp: {info.get('grasp_established', False)}"
        )
        lines.append("TABLETOP MODE: plug starts on table; pick it up manually.")

        for i, line in enumerate(lines):
            surface = self.font.render(line, True, (235, 235, 235))
            self.screen.blit(surface, (15, 10 + i * 23))

        pygame.display.flip()

    @staticmethod
    def yaw_matrix(degrees):
        a = math.radians(degrees)
        c, s = math.cos(a), math.sin(a)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def initialize_target(self):
        pos, rot = self.env.grasp_pose()
        self.target_xyz = pos.copy()
        self.target_rot = rot.copy()
        self.target_finger = float(self.env.target[7])

    def cartesian_to_action(self, dx, dy, dz, dyaw):
        dt = 1.0 / CONTROL_HZ
        speed = (self.coarse_speed if self.coarse else self.fine_speed) * self.speed_multiplier
        yaw_speed = (self.coarse_yaw if self.coarse else self.fine_yaw) * self.speed_multiplier

        self.target_xyz += np.array([dx, dy, dz], dtype=float) * speed * dt

        if dyaw:
            self.target_rot = self.yaw_matrix(
                dyaw * yaw_speed * dt
            ) @ self.target_rot

        # Broad safety workspace. IK remains responsible for joint limits.
        self.target_xyz = np.clip(
            self.target_xyz,
            [-0.85, -0.85, -0.02],
            [0.85, 0.85, 0.55],
        )

        try:
            q = self.env.solve_ik(
                self.target_xyz,
                self.target_rot,
                initial=self.env.target[:7],
            )
        except ValueError:
            # Keep the last valid robot target if the requested Cartesian pose
            # is unreachable.
            q = self.env.target[:7].copy()

        return np.r_[q, self.target_finger]

    def keyboard(self):
        requested = self.env.target.copy()
        special = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                special = "quit"

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    special = "quit"
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_TAB:
                    self.coarse = not self.coarse
                elif event.key == pygame.K_n:
                    special = "reset"
                elif event.key == pygame.K_x:
                    special = "abort"
                elif event.key == pygame.K_RETURN:
                    special = "save"
                elif event.key == pygame.K_o:
                    self.target_finger = self.env._open_travel()
                    requested[7] = self.target_finger
                elif event.key == pygame.K_c:
                    self.target_finger = 0.0
                    requested[7] = self.target_finger

        if not self.paused:
            keys = pygame.key.get_pressed()
            dx = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
            dy = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
            dz = float(keys[pygame.K_r]) - float(keys[pygame.K_f])
            dyaw = float(keys[pygame.K_e]) - float(keys[pygame.K_q])

            if dx or dy or dz or dyaw:
                requested = self.cartesian_to_action(dx, dy, dz, dyaw)

        return requested, special

    def reset_scene(self, scene):
        # IMPORTANT:
        # Use the exact same reset path as collect_manual.py:
        # collection_bank -> as_reset_options -> env.reset().
        #
        # This keeps the plug FREE on the tabletop and lets the operator
        # perform the complete pick -> lift -> transport -> insert task.
        options = as_reset_options(scene["options"])

        obs, info = self.env.reset(
            seed=int(scene["seed"]),
            options=options,
        )

        # Do not replace the tabletop options with grasped-start options.
        # Verify that the environment actually selected tabletop mode.
        if getattr(self.env, "mode", None) != "tabletop":
            raise RuntimeError(
                f"Expected OpenArmInsertEnv reset_mode='tabletop', "
                f"but got {getattr(self.env, 'mode', None)!r}. "
                "The Cartesian collector must start with the plug on the table."
            )

        # Start Cartesian control from the environment's current target.
        # This avoids using a task-specific grasp pose as the initial robot target.
        self.target_xyz, self.target_rot = self._current_target_pose()
        self.target_finger = float(self.env.target[7])
        self.paused = False

        diagnostics = dict(scene["sampler"])
        diagnostics.update({
            "id": scene["id"],
            "seed": scene["seed"],
            "split": scene["split"],
            "reset_mode": self.env.mode,
            "scene_options": options,
        })

        return obs, info, diagnostics

    def _current_target_pose(self):
        """Return the Cartesian pose represented by the current joint target.

        The collector uses the environment's current 7-joint target and asks
        MuJoCo's model/data to obtain the end-effector pose.  The fallback to
        grasp_pose() is kept only for environments that do not expose the
        expected end-effector site.
        """
        q = np.asarray(self.env.target[:7], dtype=float).copy()

        # Prefer a named end-effector site if the environment exposes one.
        site_candidates = []
        for attr in ("ee_site", "ee_site_name", "end_effector_site", "end_effector_site_name"):
            value = getattr(self.env, attr, None)
            if isinstance(value, str):
                site_candidates.append(value)

        site_candidates += ["right_gripper", "right_ee", "ee", "end_effector"]

        for name in site_candidates:
            try:
                sid = mujoco.mj_name2id(
                    self.env.model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    name,
                )
                if sid >= 0:
                    # Temporarily evaluate the current target through FK.
                    qpos_backup = self.env.data.qpos.copy()
                    qvel_backup = self.env.data.qvel.copy()
                    self.env.data.qpos[self.env.qa["right"]] = q
                    self.env.data.qvel[self.env.va["right"]] = 0.0
                    mujoco.mj_forward(self.env.model, self.env.data)

                    pos = self.env.data.site_xpos[sid].copy()
                    rot = self.env.data.site_xmat[sid].reshape(3, 3).copy()

                    self.env.data.qpos[:] = qpos_backup
                    self.env.data.qvel[:] = qvel_backup
                    mujoco.mj_forward(self.env.model, self.env.data)
                    return pos, rot
            except Exception:
                pass

        # Last-resort compatibility fallback.
        return self.env.grasp_pose()

    @staticmethod
    def append_obs(obs, observations):
        observations["state"].append(
            np.asarray(obs["state"], dtype=np.float32).copy()
        )
        observations["timestamp_s"].append(float(obs["timestamp_s"]))
        observations["scene_rgb"].append(
            np.asarray(obs["images"]["scene_rgb"], dtype=np.uint8).copy()
        )
        observations["wrist_rgb"].append(
            np.asarray(obs["images"]["wrist_rgb"], dtype=np.uint8).copy()
        )

    def collect_episode(self, episode, scene_index):
        obs, info, diagnostics = self.reset_scene(
            self.scenes[episode - 1]
        )

        observations = {
            "state": [],
            "timestamp_s": [],
            "scene_rgb": [],
            "wrist_rgb": [],
        }
        requested_actions = []
        applied_actions = []
        states = []
        velocities = []
        rewards = []
        terminated = []
        truncated = []
        durations = []
        phases = []
        infos = []

        # T+1 observations.
        self.append_obs(obs, observations)

        while self.running:
            requested, special = self.keyboard()

            if special == "quit":
                return None

            if special == "reset":
                print(f"[collector] Resetting episode {episode}.")
                return "reset"

            if special == "abort":
                print(f"[collector] Episode {episode} aborted.")
                return False

            # ENTER does not force success. The environment must report success.
            if special == "save" or info.get("outcome") == "success":
                break

            if self.paused:
                self.draw_status(episode, scene_index, info)
                time.sleep(0.02)
                continue

            states.append(self.env.data.qpos.copy())
            velocities.append(self.env.data.qvel.copy())

            next_obs, reward, term, trunc, info = self.env.step(requested)

            # launch_passive() does not automatically redraw after env.step().
            # The simulation can therefore move/collide while the 3D window
            # appears frozen. Keep the rendered scene synchronized.
            if self.viewer is not None:
                self.viewer.sync()


            requested_actions.append(np.asarray(requested, dtype=np.float64))
            applied_actions.append(
                np.asarray(info["applied_action"], dtype=np.float64)
            )
            rewards.append(float(reward))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
            durations.append(float(info["executed_dt_s"]))
            phases.append("manual_cartesian")
            infos.append(dict(info))

            self.append_obs(next_obs, observations)
            self.draw_status(episode, scene_index, info)

            if term or trunc:
                break

            time.sleep(max(0.0, 1.0 / CONTROL_HZ - 0.001))

        outcome = info.get("outcome", "unknown")

        if outcome != "success":
            print(
                f"[collector] Episode {episode} ended with "
                f"'{outcome}', so it will NOT be saved."
            )
            return False

        episode_dir = self.output / f"episode_{episode:05d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        manifest = self.env.manifest()
        manifest["action_source"] = "human_manual_cartesian_ik"
        manifest["collection_method"] = "keyboard_cartesian_xyz_gripper"
        manifest["collector"] = {
            "control_hz": CONTROL_HZ,
            "episode_limit_s": self.episode_limit_s,
            "cartesian_control": True,
            "ik_solver": "OpenArmInsertEnv.solve_ik",
            "fine_speed_m_s": self.fine_speed,
            "coarse_speed_m_s": self.coarse_speed,
            "fine_yaw_deg_s": self.fine_yaw,
            "coarse_yaw_deg_s": self.coarse_yaw,
            "speed_multiplier": self.speed_multiplier,
            "viewer_sync": True,
        }

        save_episode(
            episode_dir,
            observations=observations,
            requested=requested_actions,
            applied=applied_actions,
            states=states,
            velocities=velocities,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            durations=durations,
            phases=phases,
            infos=infos,
            manifest=manifest,
        )

        metadata_path = episode_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["action_source"] = "human_manual_cartesian_ik"
        metadata["collection_method"] = "keyboard_cartesian_xyz_gripper_tabletop_pick_insert"
        metadata["collector"] = manifest["collector"]
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print(f"[collector] Saved SUCCESS episode {episode}: {episode_dir}")
        return True

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.init_ui()

        self.viewer = mujoco.viewer.launch_passive(
            self.env.model,
            self.env.data,
        )

        try:
            episode = 1
            while episode <= self.n_episodes and self.running:
                result = self.collect_episode(
                    episode,
                    self.start_index + episode - 1,
                )

                if result is None:
                    break

                # N replays the same scene/episode number.
                if result == "reset":
                    continue

                episode += 1
        finally:
            try:
                if self.viewer is not None:
                    self.viewer.close()
            finally:
                self.env.close()
                pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Collect OpenArm demonstrations with Cartesian XYZ + gripper control."
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episode-limit",
        type=float,
        default=DEFAULT_EPISODE_LIMIT_S,
        help="Maximum simulated seconds per episode.",
    )
    parser.add_argument("--fine-speed", type=float, default=FINE_SPEED_M_S)
    parser.add_argument("--coarse-speed", type=float, default=COARSE_SPEED_M_S)
    parser.add_argument("--fine-yaw", type=float, default=FINE_YAW_DEG_S)
    parser.add_argument("--coarse-yaw", type=float, default=COARSE_YAW_DEG_S)
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=DEFAULT_SPEED_MULTIPLIER,
        help="Multiply Cartesian translation/yaw speed (default: 2.0).",
    )

    args = parser.parse_args()

    CartesianCollector(
        output=args.output,
        episodes=args.episodes,
        start=args.start,
        episode_limit_s=args.episode_limit,
        fine_speed=args.fine_speed,
        coarse_speed=args.coarse_speed,
        fine_yaw=args.fine_yaw,
        coarse_yaw=args.coarse_yaw,
        speed_multiplier=args.speed_multiplier,
    ).run()


if __name__ == "__main__":
    main()