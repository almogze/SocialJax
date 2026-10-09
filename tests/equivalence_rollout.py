"""Roll an env forward with fixed keys and random actions; print one hash per step.

Used to prove two SocialJax checkouts produce identical rollouts:
  PYTHONPATH=<checkout> JAX_PLATFORMS=cpu python tests/equivalence_rollout.py <env_id> <num_agents> <steps> <seed> > out.txt
then diff the two outputs. Each line: step, sha256 of the observation bytes, the rewards, and sha256 of every state leaf.
"""
import hashlib
import sys

import jax
import numpy as onp

import socialjax


def h(x):
    return hashlib.sha256(onp.ascontiguousarray(onp.asarray(x)).tobytes()).hexdigest()[:16]


def main(env_id, num_agents, steps, seed):
    env = socialjax.make(env_id, num_agents=num_agents, shared_rewards=False)
    key = jax.random.PRNGKey(seed)
    rng = onp.random.default_rng(seed)
    key, sub = jax.random.split(key)
    obs, state = env.reset(sub)
    step = jax.jit(env.step_env)
    n_act = env.action_space(env.agents[0]).n if hasattr(env, "agents") else 8
    print(f"reset obs={h(obs)} state={h(onp.concatenate([onp.ravel(onp.asarray(l)).astype(onp.float64) for l in jax.tree.leaves(state)]))}")
    for t in range(steps):
        key, sub = jax.random.split(key)
        acts = rng.integers(0, n_act, size=num_agents)
        obs, state, rew, done, info = step(sub, state, [jax.numpy.array(a) for a in acts])
        leaves = onp.concatenate([onp.ravel(onp.asarray(l)).astype(onp.float64) for l in jax.tree.leaves(state)])
        r = onp.asarray(rew).round(6).tolist()
        print(f"{t} obs={h(obs)} rew={r} state={h(leaves)}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
