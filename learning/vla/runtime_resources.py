"""Resource guards shared by current training and lab launchers."""
from pathlib import Path
import shutil
import subprocess

from learning.vla.inference_cadence import resource_violations


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def resources(disk_path):
    import psutil
    rows = subprocess.check_output([
        'nvidia-smi', '--query-gpu=temperature.gpu,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
    # Compose exposes one selected GPU. Refuse ambiguous multi-GPU host telemetry.
    if len(rows) != 1:
        raise ValueError('Expose exactly one GPU for resource monitoring')
    temperature, free_gpu = map(float, rows[0].split(','))
    values = {'temperature_c': temperature, 'gpu_free_mib': free_gpu,
              'ram_available_bytes': psutil.virtual_memory().available,
              'disk_free_bytes': shutil.disk_usage(existing_parent(disk_path)).free}
    violations = resource_violations(**values)
    if violations:
        raise RuntimeError(str((violations, values)))
    return values
