"""Reproducible inference cadence diagnostics, not a new control policy."""
import math


def anchor_noise_seed(base_seed, executed_commands):
    if (type(base_seed) is not int or type(executed_commands) is not int
            or min(base_seed, executed_commands) < 0 or base_seed+executed_commands >= 2**63):
        raise ValueError('Nonnegative integer seed and command index required')
    return base_seed+executed_commands


def resource_violations(*,temperature_c,gpu_free_mib,ram_available_bytes,disk_free_bytes):
    reasons=[]
    if temperature_c>=85: reasons.append('GPU temperature at/above 85 C')
    if gpu_free_mib<512: reasons.append('GPU free memory below 512 MiB')
    if ram_available_bytes<1024**3: reasons.append('Available RAM below 1 GiB')
    if disk_free_bytes<25*1024**3: reasons.append('Disk free below 25 GiB reserve')
    return reasons


def audit_cadence_chunks(chunks, *, prefix, steps, seed, schedule):
    if prefix not in (1,5,25) or steps<1 or schedule not in ('sequential','step-keyed'):
        raise ValueError('Invalid cadence protocol')
    cursor=0
    for c in chunks:
        executed=c['executed_steps']
        if (c['start_step']!=cursor or type(executed) is not int or not 1<=executed<=prefix
                or (cursor+executed<steps and executed!=prefix)):
            raise ValueError('Shifted, skipped or truncated command prefix')
        if schedule=='step-keyed' and c.get('noise_seed')!=anchor_noise_seed(seed,cursor):
            raise ValueError('Wrong common-anchor noise seed')
        cursor+=executed
    if cursor!=steps: raise ValueError('Incomplete cadence trace')
    return {'chunks':len(chunks),'commands':cursor,'prefix':prefix,'schedule':schedule}


def continuation_state(plan,failure,conditions):
    """Only continue a resource interruption; charge previous time conservatively."""
    if (plan.get('protocol')!='fixed-actor-cadence-v1' or failure.get('completed') is not False
            or failure.get('error')!="RuntimeError('GPU thermal/memory or RAM/disk safety guard')"
            or failure.get('conditions')!=conditions or plan.get('total_limit_s')!=3600):
        raise ValueError('Expected the preserved initial resource interruption')
    elapsed=failure['elapsed_s']
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<elapsed<3600:
        raise ValueError('Invalid prior time ledger')
    cases=plan['cases']
    if not 0<len(conditions)<len(cases): raise ValueError('Expected a nonempty completed prefix')
    if [{k:r[k] for k in ('episode','prefix','mode','noise_schedule')} for r in conditions]!=cases[:len(conditions)]:
        raise ValueError('Completed conditions are not the frozen case prefix')
    if failure['remaining_cases_not_run']!=cases[len(conditions):]:
        raise ValueError('Interrupted case ordering differs')
    return {'charged_prior_s':math.ceil(elapsed),'completed_conditions':conditions,
        'restart_interrupted_condition':cases[len(conditions)]}
