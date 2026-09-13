"""Execute a policy chunk through the same per-command robot API."""
import numpy as np


def execute_chunk(env, actions, *, prefix=None):
    actions=np.asarray(actions,dtype=float)
    if actions.ndim!=2 or actions.shape[1]!=8 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected a nonempty finite H x 8 action chunk")
    count=len(actions) if prefix is None else prefix
    if not isinstance(count,int) or not 1<=count<=len(actions):
        raise ValueError("Invalid execution prefix")
    transitions=[]
    for action in actions[:count]:
        transition=env.step(action)
        transitions.append(transition)
        if transition[2] or transition[3]:
            break
    return transitions

