"""Seeded bidirectional joint-space RRT with a rigid carried-plug collision proxy.

All configuration writes are scratch-only planning queries. Live execution must
use actuators and monitor the physical grasp; the proxy is not a simulation weld.
"""
import mujoco
import numpy as np


def plan_carry(env, start, goal, relative_position, relative_rotation, seed=13, iterations=2500):
    m=env.model
    scratch=mujoco.MjData(m)
    scratch.qpos[:]=env.data.qpos
    rng=np.random.default_rng(seed)
    checks=0

    def clear(q):
        nonlocal checks
        checks+=1
        scratch.qpos[env.qa['right']]=q
        mujoco.mj_forward(m,scratch)
        rotation=scratch.site_xmat[env.grasp_site].reshape(3,3)
        scratch.qpos[env.plug_qadr:env.plug_qadr+3]=scratch.site_xpos[env.grasp_site]+rotation@relative_position
        quat=np.zeros(4)
        mujoco.mju_mat2Quat(quat,(rotation@relative_rotation).ravel())
        scratch.qpos[env.plug_qadr+3:env.plug_qadr+7]=quat
        mujoco.mj_forward(m,scratch)
        for c in scratch.contact:
            bodies={int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)}
            if env.plug in bodies and bodies & set(env.fingers):continue
            if (bodies & env.robot_bodies or env.plug in bodies) and c.dist<.0005:return False
        return True

    def edge(a,b):
        count=max(1,int(np.ceil(np.max(np.abs(b-a))/.015)))
        return all(clear(a+(b-a)*t/count) for t in range(1,count+1))

    if not clear(start) or not clear(goal):raise RuntimeError('Carried-plug planner endpoint collision')
    if edge(start,goal):return [goal],{'collision_queries':checks,'nodes':2}
    trees=[([np.array(start)],[None]),([np.array(goal)],[None])]
    def extend(tree,target):
        nodes,parents=tree
        index=int(np.argmin([np.linalg.norm(q-target) for q in nodes]))
        previous=nodes[index]
        distance=np.linalg.norm(target-previous)
        q=previous+(target-previous)*min(1.,.18/max(distance,1e-12))
        if not edge(previous,q):return None
        nodes.append(q);parents.append(index)
        return len(nodes)-1
    def chain(tree,index):
        nodes,parents=tree;path=[]
        while index is not None:path.append(nodes[index]);index=parents[index]
        return path[::-1]
    for iteration in range(iterations):
        side=iteration%2;other=1-side
        target=trees[other][0][-1] if rng.random()<.25 else rng.uniform(env.lower+.005,env.upper-.005)
        a=extend(trees[side],target)
        if a is None:continue
        q=trees[side][0][a]
        for _ in range(80):
            b=extend(trees[other],q)
            if b is None:break
            if np.linalg.norm(trees[other][0][b]-q)<1e-8:
                first=chain(trees[side],a);second=chain(trees[other],b)
                path=first+second[-2::-1]
                if side==1:path=path[::-1]
                # Deterministic farthest-visible shortcutting.
                smooth=[path[0]];i=0
                while i<len(path)-1:
                    j=len(path)-1
                    while j>i+1 and not edge(path[i],path[j]):j-=1
                    smooth.append(path[j]);i=j
                return smooth[1:],{'collision_queries':checks,'nodes':sum(len(t[0]) for t in trees),
                                   'iterations':iteration+1,'waypoints':len(smooth)-1,'seed':seed}
    raise RuntimeError(f'Carry planner exhausted {iterations} iterations / {checks} collision queries')
