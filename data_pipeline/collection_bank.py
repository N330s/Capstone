"""Versioned reset banks. Evaluation conditions are never used for collection."""
import numpy as np


def collection_bank():
    # A small interpretable batch: signed lateral, distance, angle and combined starts.
    values = [
        (0, .25, -.25, 0, 0, 0), (-2, -.4, .3, 0, 0, 0),
        (2, .3, .4, 0, 0, 0), (0, -.6, -.4, 0, 0, 0),
        (0, .8, 0, 0, 0, .5), (0, -.8, 0, 0, 0, -.5),
        (-1, 0, .8, 0, .5, 0), (-1, 0, -.8, 0, -.5, 0),
        (-2, .5, -.5, .5, -.3, .3), (1, -.5, .5, -.5, .3, -.3),
    ]
    keys = ("offset_x_m", "offset_y_m", "offset_z_m", "roll_deg", "pitch_deg", "yaw_deg")
    return [{"id": f"varied_{i:04d}", "seed": 1000+i,
             "split": "validation" if i in (3,9) else "train",
             "options": dict(zip(keys, [v/1000 if j<3 else v for j,v in enumerate(row)]))}
            for i,row in enumerate(values)]


def evaluation_bank():
    rng = np.random.default_rng(90210)
    keys = ("offset_x_m", "offset_y_m", "offset_z_m", "roll_deg", "pitch_deg", "yaw_deg")
    limits = np.array([.002, .0008, .0008, .5, .5, .5])
    return [{"id": f"heldout_{i:04d}", "seed": 10000+i,
             "split": "evaluation", "options": dict(zip(keys,rng.uniform(-limits,limits).tolist()))}
            for i in range(100)]
