"""Physical-unit generated-command errors (offline probes, not task success)."""
import numpy as np


def errors(prediction, target):
    result = {}
    for n in (5, 50):
        delta = np.asarray(prediction)[:n] - np.asarray(target)[:n]
        result[str(n)] = {'arm_rmse_rad': float(np.sqrt(np.mean(delta[:, :7]**2))),
                         'finger_rmse_mm': float(1000*np.sqrt(np.mean(delta[:, 7]**2))),
                         'finger_bias_mm': float(1000*np.mean(delta[:, 7]))}
    return result
