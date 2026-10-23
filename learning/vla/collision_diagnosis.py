"""Read-only contact chronology and pose comparisons; no controller changes."""
import numpy as np


def first_index(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def orientation_error_deg(a, b):
    relative = np.asarray(a) @ np.swapaxes(np.asarray(b), -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)
    return np.degrees(np.arccos(cosine))


def contact_chronology(diagnostics, states, plug_z):
    states, plug_z = np.asarray(states), np.asarray(plug_z)
    if states.shape != (len(diagnostics), 16) or plug_z.shape != (len(diagnostics),):
        raise ValueError('Aligned T+1 diagnostics, positions and plug heights required')
    contacts = np.array([d['plug_finger_contacts'] for d in diagnostics])
    lift = (plug_z-plug_z[0])*1000
    return {'first_one_or_two_contacts':first_index(contacts>0),
        'first_two_contacts':first_index(contacts==2),
        'first_grasp':first_index([d['grasp_established'] for d in diagnostics]),
        'first_below_33mm':first_index(states[:,15]<.033),
        'first_below_30mm':first_index(states[:,15]<.030),
        'maximum_plug_lift_mm':float(lift.max()), 'maximum_lift_command':int(lift.argmax()),
        'commands_with_one_contact':int(np.sum(contacts==1)),
        'commands_with_two_contacts':int(np.sum(contacts==2))}


def nearest_path(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.ndim != 2 or reference.ndim != 2 or actual.shape[1] != 3 or reference.shape[1] != 3:
        raise ValueError('Two Cartesian point paths required')
    distance = np.linalg.norm(actual[:,None]-reference[None], axis=2)
    indices = distance.argmin(axis=1)
    return distance[np.arange(len(actual)),indices], indices
