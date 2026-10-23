"""Bounded supervision of only the process tree started by this job."""
import threading


def kill_owned_process(process):
    import psutil
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()


def arm_deadline(process, seconds, fired):
    def terminate():
        if process.poll() is None:
            fired.set()
            kill_owned_process(process)
    timer = threading.Timer(max(0, seconds), terminate)
    timer.daemon = True
    timer.start()
    return timer
