"""RealSense D435-like depth stream from rendered z-depth (chest camera, ``cameras.chest_rgbd``).

A real D435 computes depth on board by active stereo (two IR imagers 50 mm apart and an IR dot
projector) and streams z16: uint16 z-depth in millimetres, 0 = no data, aligned here to the RGB
stream. MuJoCo cannot project the IR pattern, so instead of stereo matching we render exact
z-depth from the RGB camera and degrade it with the effects that dominate a real stream at this
working distance (0.4-0.7 m):

  * RMS depth noise sigma(z) = z^2 * subpixel / (fx * baseline)   (Intel's D400 RMS-error formula)
  * no data outside [min_range, max_range]
  * holes where depth jumps between neighbouring pixels (object edges / occlusion shadows)
  * sparse random dropout
  * quantisation to 1 mm

Not modelled: IR-specific failures on dark or shiny materials, temporal/spatial filters, rolling
shutter, the colour imager's lateral offset. Deterministic for a given ``rng``.
Robot-free, so it is unit-tested without the scene.
"""
import numpy as np


def intrinsics(width, height, fovy_deg):
    """Pinhole intrinsics of a MuJoCo camera (square pixels, principal point at the centre)."""
    fy = 0.5 * height / np.tan(np.radians(fovy_deg) / 2.)
    return {"width": int(width), "height": int(height), "fx": float(fy), "fy": float(fy),
            "cx": (width - 1) / 2., "cy": (height - 1) / 2., "model": "pinhole, no distortion",
            "depth_units_m": 0.001}


def noise_sigma_m(z_m, params):
    """Intel D400 RMS depth error at distance z (m)."""
    return np.square(z_m) * params["subpixel"] / (params["depth_fx_px"] * params["baseline_m"])


def edge_holes(z_m, params):
    """Pixels next to a depth jump larger than ``edge_jump_m`` (dilated by ``edge_dilate_px``)."""
    jump = np.zeros(z_m.shape, dtype=bool)
    dx = np.abs(np.diff(z_m, axis=1)) > params["edge_jump_m"]
    dy = np.abs(np.diff(z_m, axis=0)) > params["edge_jump_m"]
    jump[:, :-1] |= dx
    jump[:, 1:] |= dx
    jump[:-1, :] |= dy
    jump[1:, :] |= dy
    for _ in range(int(params.get("edge_dilate_px", 0))):
        grown = jump.copy()
        grown[:, :-1] |= jump[:, 1:]
        grown[:, 1:] |= jump[:, :-1]
        grown[:-1, :] |= jump[1:, :]
        grown[1:, :] |= jump[:-1, :]
        jump = grown
    return jump


def degrade_depth(z_m, rng, params):
    """Exact rendered z-depth (m, float) -> D435-like z16 frame (uint16 mm, 0 = no data)."""
    z = np.asarray(z_m, dtype=np.float64)
    noisy = z + rng.standard_normal(z.shape) * noise_sigma_m(z, params)
    invalid = ((z < params["min_range_m"]) | (z > params["max_range_m"]) | ~np.isfinite(z)
               | edge_holes(z, params) | (rng.random(z.shape) < params["dropout_rate"]))
    mm = np.rint(noisy / params.get("units_m", 0.001))
    mm[invalid] = 0
    return np.clip(mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
