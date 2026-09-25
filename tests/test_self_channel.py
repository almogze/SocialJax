"""Every agent must see itself, exactly once, and never as "another agent".

combine_channels used to read the self channel at the raw grid value (k + len(Items)) instead of
the one-hot channel (k + len(Items) - 1). Agents 0..N-2 then saw agent k+1 as "self" and their own
body as "other"; only the last agent (clamped index) saw itself. That is a one-bit identity leak
to a shared, ID-free policy. This test pins the fix for every env built on combine_channels.
"""
import jax
import numpy as np
import pytest

import socialjax

# env id -> (num_agents, index of the self channel = len(Items) - 1)
CASES = {
    "gift": (9, 3),
    "clean_up": (7, 9),
    "harvest_common_open": (7, 5),
}


@pytest.mark.parametrize("env_id", sorted(CASES))
def test_each_agent_sees_itself_once(env_id):
    n, self_ch = CASES[env_id]
    env = socialjax.make(env_id, num_agents=n)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    for t in range(40):
        o = np.asarray(obs)  # (N, H, W, C)
        selfmap = o[..., self_ch] != 0
        per_agent = selfmap.reshape(n, -1).sum(1)
        assert (per_agent == 1).all(), f"{env_id} step {t}: self-channel cells per agent {per_agent.tolist()}"
        other_on_self = (selfmap & (o[..., self_ch + 1] != 0)).sum()
        assert other_on_self == 0, f"{env_id} step {t}: an agent sees itself as another agent"
        key, k1, k2 = jax.random.split(key, 3)
        acts = jax.random.randint(k1, (n,), 0, env.action_space(0).n)
        # same call the MARP wrapper makes (marp/envs/socialjax_wrapper.py)
        obs, state, _, _, _ = env.step_env(k2, state, [acts[i] for i in range(n)])
