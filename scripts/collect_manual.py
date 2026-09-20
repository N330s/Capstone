"""
Manual demonstration collector for OpenArmInsertEnv.

Controls
--------
1..7 : select right-arm joint
A/D  : decrease/increase selected joint target
Q/E  : coarse decrease/increase selected joint target
R/F  : open/close gripper target
SPACE: pause/unpause simulation stepping
S    : mark current episode successful and save it
X    : abort current episode without saving
N    : reset/restart current scene
ESC  : quit

The collector uses the same episode format as episodes.py:
T actions + T+1 observations, including state, qpos/qvel, scene/wrist RGB,
timestamps, rewards, termination flags, and diagnostics.

Joint actions are the 7 right-joint target positions in radians plus
right_finger_travel in metres.
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = "data/manual_demos"
DEFAULT_EPISODES = 10

# Fine and coarse joint increments, radians per keyboard tick.
JOINT_STEP_FINE = math.radians(0.5)
JOINT_STEP_COARSE = math.radians(2.0)

# Finger travel is in metres. The environment clips to [0, 0.044].
FINGER_STEP_FINE = 0.001
FINGER_STEP_COARSE = 0.004

CONTROL_HZ = 30.0


class Collector:
    def __init__(self, output_dir: str, n_episodes: int, start_index: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_episodes = int(n_episodes)
        self.start_index = int(start_index)

        # Images are required by episodes.py validation.
        self.env = OpenArmInsertEnv(images=True)

        # Generate collection scenes from the same deterministic scene bank.
        # workspace_for_env() centers the sampler on the actual robot mount.
        ws = workspace_for_env(self.env)
        self.bank = collection_bank(
            self.start_index + self.n_episodes,
            ws=ws,
            start=0,
        )[self.start_index:self.start_index + self.n_episodes]

        self.viewer = None
        self.running = True
        self.paused = False

        self.selected_joint = 0
        self.coarse = False

        # pygame is only used as a keyboard-control window.
        pygame.init()
        self.screen = pygame.display.set_mode((620, 260))
        pygame.display.set_caption("OpenArm manual data collector")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # Viewer / reset
    # ------------------------------------------------------------------

    def reset_scene(self, row):
        options = as_reset_options(row["options"])
        obs, info = self.env.reset(seed=row["seed"], options=options)

        # Start passive MuJoCo viewer after the first reset.
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.env.model,
                self.env.data,
            )

        self.viewer.sync()

        # Manual action starts from the environment's current target.
        action = self.env.target.copy()

        return obs, info, action

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def handle_events(self, action):
        """
        Returns:
            action, command
        command is one of:
            None, "save", "abort", "reset", "quit"
        """
        command = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                command = "quit"

            elif event.type == pygame.KEYDOWN:
                key = event.key

                if key == pygame.K_ESCAPE:
                    command = "quit"

                elif pygame.K_1 <= key <= pygame.K_7:
                    self.selected_joint = key - pygame.K_1

                elif key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif key == pygame.K_s:
                    command = "save"

                elif key == pygame.K_x:
                    command = "abort"

                elif key == pygame.K_n:
                    command = "reset"

                elif key == pygame.K_TAB:
                    self.coarse = not self.coarse

                # Selected joint control.
                elif key in (pygame.K_a, pygame.K_d):
                    step = (
                        JOINT_STEP_COARSE
                        if self.coarse
                        else JOINT_STEP_FINE
                    )
                    action[self.selected_joint] += (
                        step if key == pygame.K_d else -step
                    )

                elif key in (pygame.K_q, pygame.K_e):
                    step = (
                        JOINT_STEP_COARSE
                        if self.coarse
                        else JOINT_STEP_FINE
                    ) * 4.0
                    action[self.selected_joint] += (
                        step if key == pygame.K_e else -step
                    )

                # Gripper control.
                elif key in (pygame.K_r, pygame.K_f):
                    step = (
                        FINGER_STEP_COARSE
                        if self.coarse
                        else FINGER_STEP_FINE
                    )
                    action[7] += step if key == pygame.K_r else -step

        # Respect physical action limits before passing to the environment.
        action[:7] = np.clip(action[:7], self.env.lower, self.env.upper)
        action[7] = np.clip(action[7], 0.0, 0.044)

        return action, command

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def draw_status(self, episode_no, row, action, info, saved_count):
        self.screen.fill((25, 25, 25))

        lines = [
            f"Episode: {episode_no}/{self.n_episodes}    "
            f"Scene: {row['id']}",
            f"Selected joint: J{self.selected_joint + 1}    "
            f"Mode: {'COARSE' if self.coarse else 'FINE'}    "
            f"{'PAUSED' if self.paused else 'RUNNING'}",
            "",
            "A/D = selected joint +/-     Q/E = coarse +/-",
            "R/F = gripper close/open     TAB = fine/coarse",
            "1..7 = select joint          SPACE = pause",
            "S = save SUCCESS              X = abort (no save)",
            "N = reset scene               ESC = quit",
            "",
            "Right joint targets [rad]:",
            " ".join(f"{v:+.3f}" for v in action[:7]),
            f"Finger travel [m]: {action[7]:.4f}",
            "",
            f"Outcome: {info.get('outcome', 'running')}    "
            f"grip force: {info.get('grip_force_n', 0.0):.2f} N",
            f"Saved episodes: {saved_count}",
        ]

        y = 10
        for i, line in enumerate(lines):
            font = self.font if i < 3 else self.small_font
            surf = font.render(line, True, (235, 235, 235))
            self.screen.blit(surf, (10, y))
            y += 22 if i < 3 else 20

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episode(self, episode_number, row, saved_count):
        obs, info, action = self.reset_scene(row)

        observations = [obs]
        requested = []
        applied = []
        states = []
        velocities = []
        rewards = []
        terminated = []
        truncated = []
        durations = []
        phases = []
        infos = []

        # The episode format requires qpos/qvel for each action interval.
        # Store the state preceding each action.
        done = False
        last_control_time = time.perf_counter()

        while self.running and not done:
            self.clock.tick(CONTROL_HZ)

            action, command = self.handle_events(action)

            if command == "quit":
                self.running = False
                break

            if command == "abort":
                print(f"[collector] Aborted episode {episode_number}")
                return False, saved_count

            if command == "reset":
                print(f"[collector] Restarting scene {row['id']}")
                return self.collect_episode(
                    episode_number, row, saved_count
                )

            # Save success manually. The environment itself may also terminate
            # naturally when its success hold condition is reached.
            if command == "save":
                if info.get("outcome") == "success":
                    done = True
                else:
                    print(
                        "[collector] Cannot save yet: environment outcome is "
                        f"{info.get('outcome', 'running')}. "
                        "Continue until MuJoCo reports success."
                    )
                    continue

            self.draw_status(
                episode_number, row, action, info, saved_count
            )

            if self.paused or done:
                if self.viewer is not None:
                    self.viewer.sync()
                continue

            # Observation i precedes action i.
            states.append(self.env.data.qpos.copy())
            velocities.append(self.env.data.qvel.copy())
            requested_action = action.copy()

            next_obs, reward, term, trunc, step_info = self.env.step(
                requested_action
            )

            applied_action = np.asarray(
                step_info["applied_action"], dtype=float
            ).copy()

            requested.append(requested_action)
            applied.append(applied_action)
            rewards.append(float(reward))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
            durations.append(float(step_info["executed_dt_s"]))
            phases.append(str(step_info.get("phase", "manual")))
            infos.append(step_info)
            observations.append(next_obs)

            info = step_info

            if self.viewer is not None:
                self.viewer.sync()

            done = bool(term or trunc)

            # If the environment terminated successfully, this is a valid
            # demonstration and can be exported automatically.
            if done:
                break

        if not self.running:
            return False, saved_count

        # A manual save is only valid if the environment says success.
        if not infos:
            return False, saved_count

        final_info = infos[-1]
        outcome = final_info.get("outcome", "unknown")

        if outcome != "success":
            print(
                f"[collector] Episode {episode_number} ended with "
                f"'{outcome}', so it will NOT be saved."
            )
            return False, saved_count

        # Need T+1 observations and T actions.
        if len(observations) != len(requested) + 1:
            raise RuntimeError(
                "Episode alignment error: expected T+1 observations."
            )

        episode_dir = (
            self.output_dir / f"episode_{episode_number:05d}"
        )

        manifest = self.env.manifest()
        metadata = save_episode(
            episode_dir,
            observations=observations,
            requested=requested,
            applied=applied,
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

        # save_episode() was originally written for scripted expert data.
        # Override only that metadata field for this collector.
        metadata_path = episode_dir / "metadata.json"
        metadata["action_source"] = "human_manual"
        metadata["collection_method"] = "keyboard_joint_target_control"
        metadata["collector"] = {
            "selected_joint_control": True,
            "control_hz": CONTROL_HZ,
            "joint_step_fine_rad": JOINT_STEP_FINE,
            "joint_step_coarse_rad": JOINT_STEP_COARSE,
            "finger_step_fine_m": FINGER_STEP_FINE,
            "finger_step_coarse_m": FINGER_STEP_COARSE,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        saved_count += 1

        print(
            f"[collector] SAVED {episode_dir} "
            f"({len(requested)} actions, outcome=success)"
        )

        return True, saved_count

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        saved_count = 0

        try:
            for i, row in enumerate(self.bank, start=1):
                if not self.running:
                    break

                print()
                print("=" * 70)
                print(
                    f"Episode {i}/{self.n_episodes}: "
                    f"{row['id']}  seed={row['seed']}"
                )
                print("Collect the demonstration manually.")
                print("=" * 70)

                _, saved_count = self.collect_episode(
                    i,
                    row,
                    saved_count,
                )

        finally:
            if self.viewer is not None:
                self.viewer.close()

            pygame.quit()
            self.env.close()

        print()
        print("=" * 70)
        print(f"Collection finished. Saved {saved_count} episodes.")
        print(f"Dataset: {self.output_dir.resolve()}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Collect manual OpenArm MuJoCo demonstrations."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of scenes/episodes (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting collection-bank index.",
    )

    args = parser.parse_args()

    collector = Collector(
        output_dir=args.output,
        n_episodes=args.episodes,
        start_index=args.start,
    )
    collector.run()


if __name__ == "__main__":
    main()
