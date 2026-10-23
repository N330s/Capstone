"""Encode real simulator frames as a slide-compatible, labeled H.264 video."""
from pathlib import Path


def video_title(mode):
    return {"policy": "SmolVLA closed-loop simulation", "replay": "Recorded expert command replay",
            "recorded-state": "ASSISTED SmolVLA: recorded state, live images",
            "recorded-observations": "ASSISTED SmolVLA: recorded state and images"}[mode]


def write_video(path, scene, wrist, *, mode, episode, checkpoint, outcome, fps=50):
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    if len(scene) != len(wrist) or not scene:
        raise ValueError("Synchronized nonempty camera sequences required")
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except OSError:
        font = small = ImageFont.load_default()
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 1280, 624, "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for index, (a, b) in enumerate(zip(scene, wrist)):
            canvas = Image.new("RGB", (1280, 624), "#111e30")
            canvas.paste(Image.fromarray(a).resize((640, 480)), (0, 88))
            canvas.paste(Image.fromarray(b).resize((640, 480)), (640, 88))
            draw = ImageDraw.Draw(canvas)
            title = video_title(mode)
            draw.text((20, 10), f"{title} | {checkpoint}", font=font, fill="white")
            draw.text((20, 48), f"Fixed camera | {episode}", font=small, fill="#adc4ed")
            draw.text((660, 48), "Wrist camera", font=small, fill="#adc4ed")
            timing = ("ASSISTED, not autonomous | " if mode.startswith("recorded-") else "")
            timing += f"Simulation t={index / fps:.2f}s | inference pauses omitted"
            draw.text((20, 585), timing, font=small, fill="white")
            draw.text((930, 585), f"Outcome: {outcome}", font=small, fill="#ffca8a")
            if index == len(scene) // 2:
                canvas.save(Path(path).with_suffix(".png"))
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
