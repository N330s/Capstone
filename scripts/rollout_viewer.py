"""Watch rollouts live, and keep video of the ones that go wrong.

Two independent pieces:

  LiveViewer      mujoco.viewer.launch_passive, synced every N control steps,
                  optionally paced to wall-clock so the motion is watchable.
                  Closing the window aborts the run cleanly.
  VideoRecorder   offscreen frames from a named camera, written to mp4 when
                  imageio-ffmpeg is installed, gif when only imageio is, and a
                  compressed npz of frames when neither is. Always writes a
                  sidecar json with the phase of every captured frame, so you
                  can tell which part of the motion each frame belongs to.

Both are optional: the collector runs headless without either.
"""
import json
from pathlib import Path
import time
import numpy as np


class ViewerClosed(RuntimeError):
    """Raised when the user closes the live viewer window."""


class LiveViewer:
    def __init__(self, env, *, realtime=1.0, sync_every=1, title=None):
        import mujoco.viewer
        self.env = env
        self.realtime = float(realtime)
        self.sync_every = max(1, int(sync_every))
        self.handle = mujoco.viewer.launch_passive(env.model, env.data)
        self.handle.cam.distance = 1.4
        self.handle.cam.azimuth = 135.0
        self.handle.cam.elevation = -25.0
        self.count = 0
        self.wall0 = time.time()
        self.sim0 = float(env.data.time)
        self.title = title

    def episode(self, label):
        """Re-zero the pacing clock at the start of each episode."""
        self.title = label
        self.wall0, self.sim0 = time.time(), float(self.env.data.time)

    def sync(self):
        if not self.handle.is_running():
            raise ViewerClosed("viewer window closed")
        self.count += 1
        if self.count % self.sync_every:
            return
        self.handle.sync()
        if self.realtime > 0:
            sim_elapsed = (float(self.env.data.time) - self.sim0) / self.realtime
            lag = sim_elapsed - (time.time() - self.wall0)
            if lag > 0:
                time.sleep(min(lag, 0.1))

    def close(self):
        try:
            self.handle.close()
        except Exception:
            pass


class VideoRecorder:
    def __init__(self, env, *, camera="scene_rgb", height=480, width=640,
                 stride=2, fps=25, max_frames=3000):
        import mujoco
        self.env = env
        self.camera, self.stride, self.fps = camera, max(1, int(stride)), fps
        self.max_frames = max_frames
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0
        self.frames, self.phases, self.count = [], [], 0

    def capture(self, phase=None):
        self.count += 1
        if self.count % self.stride or len(self.frames) >= self.max_frames:
            return
        self.renderer.update_scene(self.env.data, camera=self.camera, scene_option=self.option)
        self.frames.append(self.renderer.render().copy())
        self.phases.append({"frame": len(self.frames) - 1,
                            "sim_time_s": round(float(self.env.data.time), 4),
                            "phase": phase})

    def reset(self):
        self.frames, self.phases, self.count = [], [], 0

    def save(self, path, metadata=None):
        """Returns the path actually written, or None when there is nothing."""
        if not self.frames:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stack = np.asarray(self.frames, dtype=np.uint8)
        written = None
        try:
            import imageio.v2 as imageio
            try:
                imageio.mimwrite(path.with_suffix(".mp4"), stack, fps=self.fps,
                                 codec="libx264", quality=7)
                written = path.with_suffix(".mp4")
            except Exception:
                imageio.mimwrite(path.with_suffix(".gif"), stack[::2], duration=2.0 / self.fps)
                written = path.with_suffix(".gif")
        except ImportError:
            np.savez_compressed(path.with_suffix(".frames.npz"), frames=stack)
            written = path.with_suffix(".frames.npz")
        sidecar = {"video": written.name, "camera": self.camera, "fps": self.fps,
                   "stride": self.stride, "frames": self.phases}
        if metadata:
            sidecar["episode"] = metadata
        path.with_suffix(".video.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return written

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass